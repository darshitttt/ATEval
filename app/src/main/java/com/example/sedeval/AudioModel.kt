package com.example.sedeval

import android.content.Context

interface AudioModel {
    fun load(context: Context, modelName: String): Boolean
    fun predict(audioFeatures: FloatArray): Any
    fun close()
}