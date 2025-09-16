package com.example.sedeval

import android.content.Intent
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import com.google.android.material.textfield.TextInputLayout

class MainActivity : AppCompatActivity() {

    private lateinit var evaluationManager: EvaluationManager
    private lateinit var modelSpinner: Spinner
    private lateinit var datasetButton: Button
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var statusTextView: TextView
    private lateinit var datasetPathTextView: TextView
    private lateinit var evaluationModeSwitch: Switch
    private lateinit var frameSizeLayout: TextInputLayout
    private lateinit var hopLengthLayout: TextInputLayout

    private var selectedDatasetUri: Uri? = null

    companion object {
        const val DATASET_REQUEST_CODE = 100
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        evaluationManager = EvaluationManager(this)

        // Initialize UI components
        modelSpinner = findViewById(R.id.spinnerModel)
        datasetButton = findViewById(R.id.buttonSelectDataset)
        startButton = findViewById(R.id.buttonStartEvaluation)
        stopButton = findViewById(R.id.buttonStopEvaluation)
        progressBar = findViewById(R.id.progressBar)
        statusTextView = findViewById(R.id.textViewStatus)
        datasetPathTextView = findViewById(R.id.textViewDatasetPath)
        evaluationModeSwitch = findViewById(R.id.evaluationModeSwitch)
        frameSizeLayout = findViewById(R.id.textInputLayoutFrameSize)
        hopLengthLayout = findViewById(R.id.textInputLayoutHopLength)

        // Set up model spinner
        val models = assets.list("")?.filter { it.endsWith(".tflite") || it.endsWith(".onnx") }
        if (models != null && models.isNotEmpty()) {
            val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, models)
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            modelSpinner.adapter = adapter
        } else {
            Toast.makeText(this, "No .tflite or .onnx models found in assets.", Toast.LENGTH_LONG).show()
        }

        // Set up UI listeners
        datasetButton.setOnClickListener {
            openFolderPicker()
        }

        startButton.setOnClickListener {
            val selectedModel = modelSpinner.selectedItem as? String
            if (selectedModel != null && selectedDatasetUri != null) {
                val isFramewise = evaluationModeSwitch.isChecked
                val frameSizeMs = if (isFramewise) frameSizeLayout.editText?.text.toString().toIntOrNull() ?: 1000 else 0
                val hopLengthMs = if (isFramewise) hopLengthLayout.editText?.text.toString().toIntOrNull() ?: 500 else 0

                //evaluationManager.startEvaluation(selectedModel, selectedDatasetUri!!, isFramewise, frameSizeMs, hopLengthMs)
                evaluationManager.startEvaluation(selectedModel, selectedDatasetUri!!)
            } else {
                Toast.makeText(this, "Please select both a model and a dataset folder.", Toast.LENGTH_SHORT).show()
            }
        }

        stopButton.setOnClickListener {
            evaluationManager.stopEvaluation()
        }

        evaluationModeSwitch.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                frameSizeLayout.visibility = View.VISIBLE
                hopLengthLayout.visibility = View.VISIBLE
            } else {
                frameSizeLayout.visibility = View.GONE
                hopLengthLayout.visibility = View.GONE
            }
        }

        // Observe evaluation status
        evaluationManager.evaluationProgress.observe(this, { progress ->
            progressBar.visibility = View.VISIBLE
            progressBar.progress = progress
            if (progress == 100) {
                progressBar.visibility = View.GONE
                startButton.visibility = View.VISIBLE
                stopButton.visibility = View.GONE
            }
        })

        evaluationManager.evaluationResults.observe(this, { result ->
            statusTextView.text = result
        })
    }

    private fun openFolderPicker() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).apply {
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
        }
        startActivityForResult(intent, DATASET_REQUEST_CODE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == DATASET_REQUEST_CODE && resultCode == RESULT_OK) {
            data?.data?.also { uri ->
                selectedDatasetUri = uri
                datasetPathTextView.text = "Dataset: ${uri.lastPathSegment}"
                contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
        }
    }
}
