package com.example.sedeval

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Environment
import android.util.Log
import androidx.lifecycle.MutableLiveData
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import androidx.documentfile.provider.DocumentFile
import kotlin.math.min
import kotlin.math.roundToInt
import com.google.gson.GsonBuilder

class EvaluationManager(private val context: Context) {

    private var currentModel: AudioModel? = null
    private var evaluationJob: Job? = null
    private var isEvaluating = false

    val evaluationResults = MutableLiveData<String>()
    val evaluationProgress = MutableLiveData<Int>() // e.g., percentage
    private val TARGET_SAMPLE_RATE = 16000

    fun startEvaluation(modelName: String, datasetUri: Uri) {
        if (isEvaluating) {
            evaluationResults.postValue("Evaluation already in progress.")
            return
        }

        evaluationJob = CoroutineScope(Dispatchers.IO).launch {
            isEvaluating = true
            evaluationResults.postValue("Starting evaluation for $modelName on dataset: ${datasetUri.lastPathSegment}...")
            evaluationProgress.postValue(0)

            try {
                // 1. Load the model
                currentModel = getModelInstance(modelName)
                if (currentModel == null || !currentModel!!.load(context, modelName)) {
                    evaluationResults.postValue("Failed to load model: $modelName. Check logs.")
                    isEvaluating = false
                    return@launch
                }
                //evaluationResults.postValue("Model '$modelName' loaded successfully. Expected inpu size: ${currentModel!!.getExpectedInputFeatureSize()}")

                //val expectedFeatureSize = currentModel!!.getExpectedInputFeatureSize()
                /*if (expectedFeatureSize <= 0) {
                    evaluationResults.postValue("Error: Model input feature size could not be determined or is invalid.")
                    isEvaluating = false
                    currentModel?.close()
                    return@launch
                }*/

                // 2. Prepare dataset
                val audioFiles = mutableListOf<Uri>()
                val labels = mutableMapOf<String, String>() // Map audio filename to ground truth label

                val documentFile = DocumentFile.fromTreeUri(context, datasetUri)
                if (documentFile != null && documentFile.isDirectory) {
                    for (file in documentFile.listFiles()) {
                        val fileName = file.name
                        if (fileName != null && (fileName.endsWith(".wav", true) || fileName.endsWith(".mp3", true))) {
                            audioFiles.add(file.uri)
                            // Basic assumption: label file has same name but .txt extension
                            val labelFileName = fileName.substringBeforeLast(".") + ".txt"
                            val labelFile = documentFile.findFile(labelFileName)
                            if (labelFile != null) {
                                context.contentResolver.openInputStream(labelFile.uri)?.bufferedReader().use { reader ->
                                    val labelContent = reader?.readLine()?.trim()
                                    if (!labelContent.isNullOrEmpty()) {
                                        labels[fileName] = labelContent
                                    } else {
                                        Log.w("Evaluation", "Empty or no content in label file for $fileName. Assigning 'unknown'.")
                                        labels[fileName] = "unknown"
                                    }
                                }
                            }
                            /*else {
                                Log.w("Evaluation", "No label file found for $fileName. Assigning 'unknown'.")
                                labels[fileName] = "unknown"
                            }*/
                        }
                    }
                }

                if (audioFiles.isEmpty()) {
                    evaluationResults.postValue("No supported audio files (.wav, .mp3) found in the dataset folder.")
                    isEvaluating = false
                    currentModel?.close()
                    return@launch
                }

                evaluationResults.postValue("Found ${audioFiles.size} audio files. Starting inference...")

                val allPerSecondPredictions = mutableListOf<Map<String, Any>>()
                val overallPredictionsForMetrics = mutableListOf<Pair<String, String>>()

                val allPredictions = mutableListOf<Pair<String, String>>() // (Ground Truth, Predicted)
                val totalFiles = audioFiles.size
                var processedCount = 0

                // 3. Iterate through audio files, run inference, collect predictions
                for (audioUri in audioFiles) {
                    if (!isEvaluating) { // Check if stop button was pressed
                        evaluationResults.postValue("Evaluation stopped by user.")
                        break
                    }

                    val currentFileName = audioUri.lastPathSegment ?: "unknown_file"
                    evaluationResults.postValue("Processing: $currentFileName (${processedCount + 1}/${totalFiles})...")

                    // --- Audio Decoding and Feature Extraction ---
                    //val decodedAudioInfo = decodeAudioFile(context, audioUri)
                    //val pcmData = decodeAudioFile(context, audioUri)
                    val decodedAudioInfo = decodeAudioFile(context, audioUri)
                    val pcmData = decodedAudioInfo?.first
                    val originalSampleRate = decodedAudioInfo?.second

                    if (pcmData == null || pcmData.isEmpty() || originalSampleRate == null || originalSampleRate <= 0) {
                        Log.e(
                            "EvaluationManager",
                            "Failed to decode audio file: $currentFileName. Skipping."
                        )
                        evaluationResults.postValue("Failed to decode: $currentFileName. Skipping.")
                        processedCount++
                        evaluationProgress.postValue((processedCount * 100 / totalFiles))
                        continue
                    }

                    // --- Audio Pre-processing (Conceptual) ---
                    // This is highly model-dependent. In a real app, you would:
                    // 1. Read audio data from 'audioUri' (e.g., using MediaExtractor or AudioRecord)
                    // 2. Decode raw audio (e.g., to PCM float)
                    // 3. Extract features (e.g., Mel-spectrogram, MFCCs, etc., as expected by your model)
                    //val audioInputFeatures = FloatArray(1000) { Math.random().toFloat() } // Placeholder: dummy features!
                    val resampledPcmData =
                        resampleAudio(pcmData, originalSampleRate, TARGET_SAMPLE_RATE)
                    val samplesPerSecondResampled = TARGET_SAMPLE_RATE
                    val totalSeconds =
                        (resampledPcmData.size.toFloat() / samplesPerSecondResampled).toInt()

                    val filePredictions =
                        mutableListOf<Map<String, Any>>() // Predictions for the current file

                    for (secondIndex in 0 until totalSeconds) {
                        if (!isEvaluating) {
                            evaluationResults.postValue("Evaluation stopped by the user")
                            break
                        }

                        val startSample = secondIndex * samplesPerSecondResampled
                        val endSample =
                            min(startSample + samplesPerSecondResampled, resampledPcmData.size)

                        //Extract the 1-second segment from the Resampled PCM data
                        val segmentPcm = resampledPcmData.sliceArray(startSample until endSample)

                        // Run Inference
                        val rawPredictions = currentModel!!.predict(segmentPcm)
                        val predictedLabels =
                            postProcessPredictions(rawPredictions).take(10) // Convert scores to class label
                        //val predictedLabels = postProcessPredictions(rawPredictions).toString() // Convert scores to class label

                        // Store per-second prediction
                        filePredictions.add(
                            mapOf(
                                "second" to secondIndex,
                                "predictions" to predictedLabels
                            )
                        )

                        if (secondIndex == 0) {
                            val groundTruthLabel = labels[currentFileName] ?: "unknown"
                            val topPredictedLabel =
                                predictedLabels.firstOrNull()?.label ?: "unknown"
                            overallPredictionsForMetrics.add(
                                Pair(
                                    groundTruthLabel,
                                    topPredictedLabel
                                )
                            )
                        }
                    }

                    allPerSecondPredictions.add(
                        mapOf(
                            "filename" to currentFileName,
                            "total_seconds" to totalSeconds,
                            "segments" to filePredictions
                        )
                    )

                    processedCount++
                    evaluationProgress.postValue((processedCount * 100 / totalFiles))
                }

                /*    // --- Run Inference ---


                    val groundTruthLabel = labels[currentFileName] ?: "unknown"
                    allPredictions.add(Pair(groundTruthLabel, predictedLabels) as Pair<String, String>)

                    processedCount++
                    evaluationProgress.postValue((processedCount * 100 / totalFiles))
                }*/

                if (isEvaluating) { // Only report if not stopped by user
                    // 4. Calculate Metrics
                    //val metrics = calculateMetrics(allPredictions)
                    val metrics = emptyMap<String, Float>()

                    // 5. Report and Save Results
                    reportResults(metrics)
                    //saveResults(allPredictions, metrics)
                    saveResults(allPerSecondPredictions, metrics)
                }

            } catch (e: Exception) {
                Log.e("EvaluationManager", "Evaluation failed: ${e.message}", e)
                evaluationResults.postValue("Evaluation failed: ${e.message}. See logs for details.")
            } finally {
                currentModel?.close()
                currentModel = null
                isEvaluating = false
                evaluationProgress.postValue(100) // Ensure progress is 100% on completion/failure
            }
        }
    }

    private fun decodeAudioFile(context: Context, audioUri: Uri): Pair<FloatArray, Int>? {
        var extractor: MediaExtractor? = null
        var decoder: MediaCodec? = null
        try {
            extractor = MediaExtractor()
            context.contentResolver.openFileDescriptor(audioUri, "r")?.use { pfd ->
                extractor.setDataSource(pfd.fileDescriptor)
            } ?: run {
                Log.e("AudioDecoder", "Failed to open asset file descriptor for URI: $audioUri")
                return null
            }

            var audioTrackIndex = -1
            for (i in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME)
                if (mime?.startsWith("audio/") == true) {
                    audioTrackIndex = i
                    break
                }
            }

            if (audioTrackIndex == -1) {
                Log.e("AudioDecoder", "No audio track found in file: $audioUri")
                return null
            }

            extractor.selectTrack(audioTrackIndex)
            val format = extractor.getTrackFormat(audioTrackIndex)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: return null
            val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

            decoder = MediaCodec.createDecoderByType(mime)
            decoder.configure(format, null, null, 0)
            decoder.start()

            val decodedAudio = mutableListOf<Float>()
            val bufferInfo = MediaCodec.BufferInfo()
            var sawInputEOS = false
            var sawOutputEOS = false
            val TIMEOUT_US: Long = 10000 // 10ms timeout

            while (!sawOutputEOS) {
                if (!sawInputEOS) {
                    val inputBufferId = decoder.dequeueInputBuffer(TIMEOUT_US)
                    if (inputBufferId >= 0) {
                        val inputBuffer = decoder.getInputBuffer(inputBufferId) ?: continue
                        val sampleSize = extractor.readSampleData(inputBuffer, 0)
                        if (sampleSize < 0) {
                            sawInputEOS = true
                            decoder.queueInputBuffer(inputBufferId, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        } else {
                            decoder.queueInputBuffer(inputBufferId, 0, sampleSize, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val outputBufferId = decoder.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)
                if (outputBufferId >= 0) {
                    val outputBuffer = decoder.getOutputBuffer(outputBufferId) ?: continue
                    outputBuffer.rewind() // Ensure buffer is at the beginning

                    // Assuming PCM 16-bit signed integer output, convert to float
                    // This conversion might need adjustment based on actual MediaCodec output format
                    val shortBuffer = outputBuffer.asShortBuffer()
                    while (shortBuffer.hasRemaining()) {
                        // Normalize 16-bit PCM to float range [-1.0, 1.0]
                        decodedAudio.add(shortBuffer.get().toFloat() / 32768.0f)
                    }
                    decoder.releaseOutputBuffer(outputBufferId, false)
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        sawOutputEOS = true
                    }
                } else if (outputBufferId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val newFormat = decoder.outputFormat
                    Log.d("AudioDecoder", "Output format changed: $newFormat")
                    // Handle format changes if necessary (e.g., sample rate, channel count)
                }
            }
            return Pair(decodedAudio.toFloatArray(), sampleRate)
        } catch (e: Exception) {
            Log.e("AudioDecoder", "Error decoding audio file: ${e.message}", e)
            return null
        } finally {
            extractor?.release()
            decoder?.stop()
            decoder?.release()
        }
    }

    private fun resampleAudio(pcmData: FloatArray, originalSampleRate: Int, targetSampleRate: Int): FloatArray {
        if (originalSampleRate == targetSampleRate) {
            return pcmData // No resampling needed
        }

        val originalLength = pcmData.size
        val targetLength = (originalLength * targetSampleRate.toFloat() / originalSampleRate).roundToInt()
        val resampledData = FloatArray(targetLength)

        for (i in 0 until targetLength) {
            val originalIndexFloat = i * (originalLength.toFloat() - 1) / (targetLength.toFloat() - 1)
            val indexFloor = originalIndexFloat.toInt()
            val indexCeil = indexFloor + 1

            if (indexCeil >= originalLength) {
                resampledData[i] = pcmData[indexFloor]
            } else {
                val ratio = originalIndexFloat - indexFloor
                resampledData[i] = pcmData[indexFloor] * (1 - ratio) + pcmData[indexCeil] * ratio
            }
        }
        return resampledData
    }

    fun stopEvaluation() {
        if (isEvaluating) {
            evaluationJob?.cancel() // Cancel the coroutine
            isEvaluating = false
            evaluationResults.postValue("Evaluation stopping initiated...")
            Log.d("EvaluationManager", "Stop initiated.")
        } else {
            evaluationResults.postValue("No evaluation currently in progress.")
        }
    }

    private fun postProcessPredictions(rawPredictions: Any): MutableList<AudioTag> {
        // Example: Get the class with the highest score
        // Replace with your actual class labels in the correct order
        val tags = mutableListOf<AudioTag>()
        val classLabels = getAudioLabels()
        val predictions = rawPredictions as Array<FloatArray>
        Log.d("EvaluationManager", "${predictions[0].size}")

        predictions[0].forEachIndexed{index, confidence ->
            if (index < classLabels.size && confidence>0.05f) {
                tags.add(AudioTag(classLabels[index], confidence))
                //Log.d("Evaluation", "Preds ${confidence}, ${index}")
            }
        }
        tags.sortByDescending { it.confidence }
        //Log.d("Preds", "$tags")

        return tags
    }

    private fun calculateMetrics(predictions: List<Pair<String, String>>): Map<String, Float> {
        if (predictions.isEmpty()) return emptyMap()

        val uniqueLabels = predictions.flatMap { listOf(it.first, it.second) }
            .toSet()
            .sorted()

        // Initialize confusion matrix
        val confusionMatrix = mutableMapOf<String, MutableMap<String, Int>>()
        for (actual in uniqueLabels) {
            val row = mutableMapOf<String, Int>()
            for (predicted in uniqueLabels) {
                row[predicted] = 0
            }
            confusionMatrix[actual] = row
        }

        // Fill confusion matrix
        for ((actual, predicted) in predictions) {
            if (actual in confusionMatrix && predicted in confusionMatrix[actual]!!) {
                confusionMatrix[actual]!![predicted] = confusionMatrix[actual]!![predicted]!! + 1
            }
        }

        // Compute accuracy
        val totalCorrect = predictions.count { it.first == it.second }
        val accuracy = totalCorrect.toFloat() / predictions.size

        val precisionScores = mutableListOf<Float>()
        val recallScores = mutableListOf<Float>()
        val f1Scores = mutableListOf<Float>()

        for (label in uniqueLabels) {
            val tp: Float = confusionMatrix[label]?.get(label)?.toFloat() ?: 0f

            // False positives: predicted 'label' when it was something else
            val fp: Float = uniqueLabels
                .filter { it != label }
                .map { other -> confusionMatrix[other]?.get(label)?.toFloat() ?: 0f }
                .sum()

            // False negatives: actual was 'label', but predicted something else
            val fn: Float = uniqueLabels
                .filter { it != label }
                .map { other -> confusionMatrix[label]?.get(other)?.toFloat() ?: 0f }
                .sum()

            val precision: Float = if ((tp + fp) > 0f) tp / (tp + fp) else 0f
            val recall: Float = if ((tp + fn) > 0f) tp / (tp + fn) else 0f
            val f1: Float = if ((precision + recall) > 0f) 2 * (precision * recall) / (precision + recall) else 0f

            precisionScores.add(precision)
            recallScores.add(recall)
            f1Scores.add(f1)
        }

        // Macro averages
        val macroPrecision = precisionScores.average().toFloat()
        val macroRecall = recallScores.average().toFloat()
        val macroF1 = f1Scores.average().toFloat()

        return mapOf(
            "Accuracy" to accuracy,
            "Precision (Macro)" to macroPrecision,
            "Recall (Macro)" to macroRecall,
            "F1 Score (Macro)" to macroF1
        )
    }

    private fun reportResults(metrics: Map<String, Float>) {
        val stringBuilder = StringBuilder("Evaluation Complete!\n\n")
        metrics.forEach { (key, value) ->
            stringBuilder.append("$key: %.4f\n".format(value))
        }
        evaluationResults.postValue(stringBuilder.toString())
    }

    private fun saveResults(perSecondPredictions: List<Map<String, Any>>, metrics: Map<String, Float>) {
        val timestamp = System.currentTimeMillis()
        val filename = "evaluation_results_$currentModel.json"

        try {
            // Get app-specific external storage directory (recommended over Downloads for app data)
            // Or use MediaStore for more structured access if files are user-facing.
            val appSpecificDir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            val file = File(appSpecificDir, filename)
            file.createNewFile()

            val gson = GsonBuilder().setPrettyPrinting().create()

            val resultsMap = mapOf(
                "overall_metrics" to metrics,
                "per_second_predictions" to perSecondPredictions
            )

            FileOutputStream(file).bufferedWriter().use { writer ->
                gson.toJson(resultsMap, writer)
            }

            /*FileOutputStream(file).bufferedWriter().use { writer ->
                writer.append("--- Evaluation Metrics ---\n")
                metrics.forEach { (key, value) ->
                    writer.append("$key: %.4f\n".format(value))
                }
                writer.append("\n--- Predictions (Ground Truth, Predicted) ---\n")
                predictions.forEach { (gt, pred) ->
                    writer.append("$gt, $pred\n")
                }
            }*/
            val savePath = file.absolutePath
            Log.d("SaveResults", "Results saved to: $savePath")
            evaluationResults.postValue(evaluationResults.value + "\nResults saved to: $savePath")
        } catch (e: Exception) {
            Log.e("SaveResults", "Error saving results: ${e.message}", e)
            evaluationResults.postValue(evaluationResults.value + "\nError saving results: ${e.message}")
        }
    }

    private fun getModelInstance(modelName: String): AudioModel? {
        return if (modelName.endsWith(".tflite")) {
            TFLiteAudioModel(useGPU = false)
        } else if (modelName.endsWith(".onnx")) {
            ONNXAudioModel()
        } else {
            null
        }
    }
}
