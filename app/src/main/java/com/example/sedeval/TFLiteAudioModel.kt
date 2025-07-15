package com.example.sedeval

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class TFLiteAudioModel : AudioModel {
    private var interpreter: Interpreter? = null

    override fun load(context: Context, modelName: String): Boolean {
        try {
            // Get resource ID from model name (e.g., "model_v1.tflite" -> R.raw.model_v1)
            /*val resourceId = context.resources.getIdentifier(modelName.split(".")[0], "raw", context.packageName)
            if (resourceId == 0) {
                Log.e("TFLiteModel", "Resource ID for $modelName not found.")
                return false
            }*/

            //val assetFileDescriptor = context.resources.openRawResourceFd(resourceId)
            val inputStream = context.assets.open(modelName)
            //val fileChannel = (inputStream as FileInputStream).channel

            val modelBytes = inputStream.readBytes()
            inputStream.close()
            val mappedByteBuffer = ByteBuffer.allocateDirect(modelBytes.size)
            mappedByteBuffer.put(modelBytes)
            mappedByteBuffer.flip()

            /*val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)*/

            val options = Interpreter.Options()
            options.setNumThreads(4) // Example: Set number of threads for inference
            interpreter = Interpreter(mappedByteBuffer, options)
            Log.d("TFLiteModel", "TFLite model loaded: $modelName")
            return true
        } catch (e: Exception) {
            Log.e("TFLiteModel", "Error loading TFLite model: ${e.message}", e)
            return false
        }
    }

    override fun predict(audioFeatures: FloatArray): FloatArray {
        // Placeholder for actual TFLite inference logic.
        // This example assumes a single-input, single-output model.
        // You would need to determine the exact input and output tensor shapes.
        val inputBuffer = ByteBuffer.allocateDirect(audioFeatures.size * 4).order(ByteOrder.nativeOrder())
        inputBuffer.asFloatBuffer().put(audioFeatures)

        // Assuming a fixed number of output classes for demonstration
        val NUM_CLASSES = 10
        val outputScores = Array(1) { FloatArray(NUM_CLASSES) } // Assuming batch size 1

        try {
            interpreter?.run(inputBuffer, outputScores)
        } catch (e: Exception) {
            Log.e("TFLiteModel", "Error running TFLite inference: ${e.message}", e)
            // Return an empty or error-indicating array
            return FloatArray(NUM_CLASSES)
        }
        return outputScores[0]
    }

    override fun close() {
        interpreter?.close()
        interpreter = null
        Log.d("TFLiteModel", "TFLite interpreter closed. ")
    }

}