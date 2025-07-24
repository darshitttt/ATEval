package com.example.sedeval // Updated package name as requested

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil // Needed for loadLabels and loadMappedFile
import java.io.IOException
import java.nio.MappedByteBuffer // FileUtil.loadMappedFile returns MappedByteBuffer

/**
 * Executes the YAMNet TensorFlow Lite model for audio classification.
 * This class encapsulates the TFLite interpreter setup, model loading, and inference logic,
 * including post-processing to extract top classification scores and labels.
 *
 * @param context The Android application context, used for accessing assets.
 * @param useGPU A boolean indicating whether to attempt to use GPU acceleration.
 * Note: As per original YamnetModelExecutor, GPU might not be supported for this model.
 */
class TFLiteAudioModel(
    //private val context: Context, // Changed to private val to be accessible in init
    private var useGPU: Boolean // This parameter is kept for consistency but not actively used in the provided logic
): AudioModel {

    // Use of 2 threads after benchmarking the model
    // Also no GPU usage because it does not support the model
    private var numberThreads = 2
    private var interpreter: Interpreter? = null // Made nullable to match close() logic
    private var predictTime = 0L
    private lateinit var labels: List<String>

    override fun load(context: Context, modelName: String): Boolean {
        try {
            // Load the YAMNet model from the assets folder
            val modelBuffer: MappedByteBuffer = FileUtil.loadMappedFile(context, modelName)
            val tfliteOptions = Interpreter.Options()
            tfliteOptions.setNumThreads(numberThreads)
            // If useGPU was true and a GPU delegate was available, it would be set here.
            // For simplicity and based on the original comment, we're not adding GPU delegate setup here.
            interpreter = Interpreter(modelBuffer, tfliteOptions)
            Log.d(TAG, "TFLite Interpreter initialized with model: $modelName")

            // Load the labels (class names) from the assets folder
            labels = FileUtil.loadLabels(context, LABELS_FILE)
            Log.d(TAG, "Labels loaded from: $LABELS_FILE")

        } catch (e: IOException) {
            Log.e(TAG, "Error loading TFLite model or labels from assets: ${e.message}", e)
            // It's good practice to re-throw or handle this more gracefully,
            // perhaps by setting a flag that the executor is not ready.
            throw IllegalStateException("Failed to initialize YamnetModelExecutor: ${e.message}", e)
        } catch (e: Exception) {
            Log.e(TAG, "General error during YamnetModelExecutor initialization: ${e.message}", e)
            throw IllegalStateException("Failed to initialize YamnetModelExecutor: ${e.message}", e)
        }
        return true
    }

    companion object {
        private const val TAG = "YamnetModelExecutor"
        private const val YAMNET_MODEL = "yamnet_oneOutput_withMetadata.tflite" // Your TFLite model file name
        private const val LABELS_FILE = "classes.txt" // Your labels file name
    }

    /**
     * Executes the YAMNet model inference on the provided audio features.
     *
     * @param floatsInput A FloatArray containing the pre-processed audio data (e.g., 2 seconds of 16kHz audio normalized to -1.0 to 1.0).
     * @return A Pair where the first element is an ArrayList of top predicted labels (String)
     * and the second element is an ArrayList of their corresponding scores (Float).
     */
    override fun predict(floatsInput: FloatArray): Any {
        predictTime = System.currentTimeMillis()

        // Prepare inputs and outputs for the TFLite interpreter
        val inputs = arrayOf<Any>(floatsInput)
        val outputs = HashMap<Int, Any>()

        // Define the output shapes expected from the YAMNet model
        // scores(4, 521), embeddings(4, 1024), spectogram(240, 64)
        val arrayScores = Array(4) { FloatArray(521) { 0f } }
        val arrayEmbeddings = Array(4) { FloatArray(1024) { 0f } }
        val arraySpectograms = Array(240) { FloatArray(64) { 0f } }

        outputs[0] = arrayScores       // Output tensor 0: scores
        outputs[1] = arrayEmbeddings   // Output tensor 1: embeddings
        outputs[2] = arraySpectograms  // Output tensor 2: spectograms

        try {
            // Run the inference
            interpreter?.runForMultipleInputsOutputs(inputs, outputs)
        } catch (e: Exception) {
            Log.e(TAG, "Error running TFLite inference: ${e.message}", e)
            //return Pair(ArrayList(), ArrayList()) // Return empty lists on error
            //return arrayListOf()
        }

        // Post-processing: Calculate mean scores across the 4 frames
        val arrayMeanScores = FloatArray(521) { 0f }
        for (i in 0 until 521) {
            // Find the average of the 4 scores for each class
            arrayMeanScores[i] = arrayListOf(
                arrayScores[0][i],
                arrayScores[1][i],
                arrayScores[2][i],
                arrayScores[3][i]
            ).average().toFloat()
        }

        // Convert to ArrayList for easier manipulation (e.g., removing elements)
        val listOfArrayMeanScores = arrayMeanScores.toCollection(ArrayList())

        val listOfMaximumValues = arrayListOf<Float>()
        val listOfMaxIndices = arrayListOf<Int>()

        // Find the top 10 scores and their corresponding indices
        // Note: This method of finding max and removing can be inefficient for large lists
        // and might have issues with duplicate values. A more robust way is to create
        // a list of (score, index) pairs, sort it, and then take the top 10.
        val scoreIndexPairs = arrayMeanScores.mapIndexed { index, score -> Pair(score, index) }
            .sortedByDescending { it.first } // Sort by score in descending order

        // Take the top 10
        for (i in 0 until minOf(scoreIndexPairs.size, 10)) {
            listOfMaximumValues.add(scoreIndexPairs[i].first)
            listOfMaxIndices.add(scoreIndexPairs[i].second)
        }

        /*Log.i(TAG, "YAMNET_SCORES (mean): ${arrayMeanScores.contentToString()}")
        Log.i(TAG, "YAMNET_SCORES_SIZE: ${arrayMeanScores.size}")
        Log.i(TAG, "YAMNET_INDICES (top 10): $listOfMaxIndices")*/

        val finalListOfOutputs = arrayListOf<String>()
        // Map the top indices to their corresponding labels
        for (index in listOfMaxIndices) {
            if (index < labels.size) { // Basic bounds check
                finalListOfOutputs.add(labels[index])
            } else {
                Log.w(TAG, "Label index $index out of bounds for labels list (size: ${labels.size})")
                finalListOfOutputs.add("Unknown Label")
            }
        }
        //Log.d("TFLite", "$arrayMeanScores")
        //Log.d("TFLite", "${arrayMeanScores.size}")
        return arrayOf(arrayMeanScores)
    }

    /**
     * Closes the TFLite interpreter and releases resources.
     */
    override fun close() {
        interpreter?.close()
        interpreter = null
        Log.d(TAG, "TFLite interpreter closed.")
    }
}