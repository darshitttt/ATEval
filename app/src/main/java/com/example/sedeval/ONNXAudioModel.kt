package com.example.sedeval

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.ByteBuffer
import java.nio.FloatBuffer

class ONNXAudioModel : AudioModel {
    private var ortSession: OrtSession? = null
    private var ortEnv: OrtEnvironment? = null
    private var inputFeatureSize: Int = 0
    private lateinit var inputFeatureShape: LongArray

    override fun load(context: Context, modelName: String): Boolean {
        try {
            ortEnv = OrtEnvironment.getEnvironment()

            val inputStream = context.assets.open(modelName)
            val modelBytes = inputStream.readBytes()
            inputStream.close()

            val modelByteBuffer = ByteBuffer.allocateDirect(modelBytes.size)
            modelByteBuffer.put(modelBytes)
            modelByteBuffer.flip()

            val sessionOptions = OrtSession.SessionOptions()
            ortSession = ortEnv?.createSession(modelByteBuffer, sessionOptions)

            val modelInputInfo = ortSession!!.inputInfo
            Log.d("ModelInput", "model input: $modelInputInfo")
            val inputName = ortSession!!.inputNames.iterator().next()
            val inputInfo = ortSession!!.inputInfo[inputName]
            inputFeatureShape = longArrayOf(1,1,64,101)
            //inputFeatureShape = inputInfo?.info?.shape?.map{

            //}


            Log.d("ModelInfo", "modelInfo ${inputInfo?.info?.toString()}")

            Log.d("ONNXModel", "ONNX model loaded from assets: $modelName")
            return true
        } catch (e: Exception) {
            Log.e("ONNXModel", "Error loading ONNX model from assets: ${e.message}", e)
            return false
        }
    }

    override fun predict(audioFeatures: FloatArray): Any {
        if (ortSession == null || ortEnv == null) {
            Log.e("ONNXModel", "ONNX model not loaded. Cannot run prediction.")
            return FloatArray(0)
        }

        var inputTensor: OnnxTensor? = null
        var results: OrtSession.Result? = null
        try {
            val inputName = ortSession!!.inputNames.iterator().next()
            val outputName = ortSession!!.outputNames.iterator().next()

            val modelInputInfo = ortSession!!.outputInfo[outputName]
            Log.d("OutputInfo", "output info: ${modelInputInfo.toString()}")
            // This line now correctly extracts the shape from the model's metadata.
            // The `map { if (it == -1L) 1L else it }` handles dynamic batch sizes.
            //val inputShape = modelInputInfo?.typeAndShapeInfo?.asTensorInfo()?.shape?.map { if (it == -1L) 1L else it }?.toLongArray()
               // ?: longArrayOf(1, audioFeatures.size.toLong()) // Fallback, but should ideally get from model
            val inputShape = longArrayOf(1, audioFeatures.size.toLong())

            // Log the expected input shape and actual buffer size for debugging
            //Log.d("ONNXModel", "Model expects input shape: ${inputShape.joinToString()}, Provided buffer size: ${audioFeatures.size}")

            // If the model expects [1,1] and audioFeatures is 1000, this is the core issue.
            // The `audioFeatures` array MUST be prepared with the correct size.
            // For now, we'll assume the dummy data in EvaluationManager is adjusted.
            val inputBuffer = FloatBuffer.wrap(audioFeatures)

            inputTensor = OnnxTensor.createTensor(ortEnv, inputBuffer, inputShape)

            val inputs = mapOf(inputName to inputTensor)

            results = ortSession?.run(inputs)

            //val outputOnnxValue = results?.get(outputName)
            val outputOnnxValue = results?.get(0)?.value

            if (outputOnnxValue == null) {
                Log.e("ONNXModel", "Failed to get output scores or output was not FloatArray. Check model output type.")
                return FloatArray(0)
            }
            return outputOnnxValue
        } catch (e: Exception) {
            Log.e("ONNXModel", "Error running ONNX inference: ${e.message}", e)
            return FloatArray(0)
        } finally {
            inputTensor?.close()
            results?.close()
        }
    }

    override fun close() {
        ortSession?.close()
        ortEnv?.close()
        ortSession = null
        ortEnv = null
        Log.d("ONNXModel", "ONNX model resources closed.")
    }
}