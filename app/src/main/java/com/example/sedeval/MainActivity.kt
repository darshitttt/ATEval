package com.example.sedeval

import android.app.Activity
import android.content.ContentValues.TAG
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import com.example.sedeval.R
import android.os.Environment
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.Observer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
//import org.tensorflow.lite.Interpreter // Assuming you've added TensorFlow Lite dependency


// --- Main Activity Implementation ---
class MainActivity : AppCompatActivity() {

    // UI elements declarations
    private lateinit var modelSpinner: Spinner
    private lateinit var datasetPathTextView: TextView
    private lateinit var chooseFolderButton: Button
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var resultsTextView: TextView
    private lateinit var closeAppButton: Button

    // Evaluation Manager instance
    private lateinit var evaluationManager: EvaluationManager

    // Variable to hold the selected dataset URI
    private var selectedDatasetUri: Uri? = null

    // Request code for picking dataset folder
    private val PICK_DATASET_FOLDER_REQUEST_CODE = 1001

    // Required permissions for external storage access
    private val REQUIRED_PERMISSIONS = arrayOf(
        android.Manifest.permission.READ_EXTERNAL_STORAGE,
        android.Manifest.permission.WRITE_EXTERNAL_STORAGE
    )
    private val REQUEST_CODE_PERMISSIONS = 123 // Request code for permissions

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main) // Set the layout for this activity

        // Initialize UI elements by finding them by their IDs
        modelSpinner = findViewById(R.id.modelSpinner)
        datasetPathTextView = findViewById(R.id.datasetPathTextView)
        chooseFolderButton = findViewById(R.id.chooseFolderButton)
        startButton = findViewById(R.id.startButton)
        stopButton = findViewById(R.id.stopButton)
        resultsTextView = findViewById(R.id.resultsTextView)
        closeAppButton = findViewById(R.id.closeAppButton)

        // Initialize the EvaluationManager, passing the application context
        evaluationManager = EvaluationManager(applicationContext)

        // Populate the model spinner with available models
        setupModelSpinner()

        // Set up observers for LiveData from EvaluationManager to update UI
        setupEvaluationObservers()

        // Set up click listeners for buttons
        setupButtonListeners()

        // Request necessary permissions when the app starts
        requestPermissionsIfNecessary()

        // Initially disable start/stop buttons until a dataset is chosen or conditions are met
        setEvaluationButtonsEnabled(false)
    }

    /**
     * Populates the model spinner with model names found in the res/raw directory.
     * Models are assumed to be named like 'model_name.tflite' or 'model_name.onnx'.
     */
    private fun setupModelSpinner() {
        val models = getAvailableModels()
        if (models.isEmpty()) {
            Toast.makeText(this, "No models found in res/raw. Please add .tflite or .onnx models.", Toast.LENGTH_LONG).show()
        }
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, models)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        modelSpinner.adapter = adapter
    }

    /**
     * Retrieves a list of available model file names from the `res/raw` directory.
     * Filters for files ending with `.tflite` or `.onnx`.
     * @return A list of model file names (e.g., "my_model.tflite").
     */
    // In MainActivity.kt, replace the existing getAvailableModels function with this:

    private fun getAvailableModels(): List<String> {
        val modelNames = mutableListOf<String>()
        try {
            // List files directly from the assets folder
            val assetFiles = assets.list("") // List all files in the root of the assets directory

            if (assetFiles != null) {
                for (fileName in assetFiles) {
                    // Check for .tflite or .onnx extensions
                    if (fileName.endsWith(".tflite", true) || fileName.endsWith(".onnx", true)) {
                        modelNames.add(fileName)
                    }
                }
            }

            // Add some dummy models if none found for testing purposes (this part remains the same)
            if (modelNames.isEmpty()) {
                modelNames.add("example_model_A.tflite")
                modelNames.add("example_model_B.onnx")
            }

        } catch (e: Exception) {
            Log.e("ModelLoader", "Error getting models from assets: ${e.message}") // Updated log message
            Toast.makeText(this, "Error listing models: ${e.message}", Toast.LENGTH_LONG).show()
        }
        return modelNames
    }

    /**
     * Sets up observers for LiveData from the EvaluationManager to update the UI.
     */
    private fun setupEvaluationObservers() {
        // Observe evaluation results and update the resultsTextView
        evaluationManager.evaluationResults.observe(this, Observer { results ->
            resultsTextView.text = results
            // Scroll to the bottom to show latest messages
            val scrollView = resultsTextView.parent as? View
            scrollView?.post { scrollView.scrollTo(0, resultsTextView.height) }
        })

        // Observe evaluation progress (e.g., for a ProgressBar, not implemented in this UI)
        evaluationManager.evaluationProgress.observe(this, Observer { progress ->
            // You could update a ProgressBar here
            // Log.d("MainActivity", "Evaluation Progress: $progress%")
        })
    }

    /**
     * Sets up click listeners for all interactive buttons.
     */
    private fun setupButtonListeners() {
        // Listener for the "Choose Folder" button
        chooseFolderButton.setOnClickListener {
            // Intent to open the document tree (folder picker)
            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE)
            intent.addCategory(Intent.CATEGORY_DEFAULT)
            startActivityForResult(intent, PICK_DATASET_FOLDER_REQUEST_CODE)
        }

        // Listener for the "Start Evaluation" button
        startButton.setOnClickListener {
            val selectedModel = modelSpinner.selectedItem as? String
            if (selectedModel.isNullOrEmpty()) {
                Toast.makeText(this, "Please select a model.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (selectedDatasetUri != null) {
                evaluationManager.startEvaluation(selectedModel, selectedDatasetUri!!)
                setEvaluationButtonsEnabled(false) // Disable buttons while evaluating
                stopButton.isEnabled = true // Only enable stop
            } else {
                Toast.makeText(this, "Please select an evaluation dataset folder first.", Toast.LENGTH_SHORT).show()
            }
        }

        // Listener for the "Stop Evaluation" button
        stopButton.setOnClickListener {
            evaluationManager.stopEvaluation()
            // Buttons will be re-enabled by the EvaluationManager's final state update
        }

        // Listener for the "Close App" button
        closeAppButton.setOnClickListener {
            finishAndRemoveTask() // Terminates the app process
        }
    }

    /**
     * Handles the result from external activities, specifically the folder picker.
     */
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_DATASET_FOLDER_REQUEST_CODE && resultCode == Activity.RESULT_OK) {
            data?.data?.let { uri ->
                selectedDatasetUri = uri
                // Persist URI permissions so the app can access it later
                contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)

                datasetPathTextView.text = getPathFromUri(uri) // Update UI with selected path
                validateDatasetFolder(uri) // Validate the folder contents
            }
        }
    }

    /**
     * Validates the selected dataset folder for the presence of audio and label files.
     * @param uri The URI of the selected folder.
     */
    private fun validateDatasetFolder(uri: Uri) {
        var hasAudioFiles = false
        var hasLabelFiles = false
        try {
            val documentFile = DocumentFile.fromTreeUri(this, uri)
            if (documentFile != null && documentFile.isDirectory) {
                for (file in documentFile.listFiles()) {
                    val fileName = file.name
                    if (fileName != null) {
                        if (fileName.endsWith(".wav", true) || fileName.endsWith(".mp3", true)) {
                            hasAudioFiles = true
                        }
                        if (fileName.endsWith(".txt", true) || fileName.endsWith(".csv", true)) {
                            // This is a basic check. Real validation might parse content to confirm labels.
                            hasLabelFiles = true
                        }
                    }
                    if (hasAudioFiles && hasLabelFiles) break // Found both, no need to continue
                }
            }
        } catch (e: Exception) {
            Log.e("DatasetValidation", "Error validating dataset folder: ${e.message}", e)
            Toast.makeText(this, "Error validating folder: ${e.message}", Toast.LENGTH_LONG).show()
        }

        if (hasAudioFiles || hasLabelFiles) {
            Toast.makeText(this, "Dataset folder valid. Ready for evaluation.", Toast.LENGTH_SHORT).show()
            setEvaluationButtonsEnabled(true) // Enable Start button
        }
        /**else {
            Toast.makeText(this, "Selected folder needs audio files (.wav/.mp3) AND label files (.txt/.csv).", Toast.LENGTH_LONG).show()
            setEvaluationButtonsEnabled(false) // Disable Start button
        }**/
    }

    /**
     * Helper function to get a displayable path from a content URI.
     * Note: This might not be a direct file system path but a user-friendly representation.
     * @param uri The content URI to convert.
     * @return A string representation of the URI path.
     */
    private fun getPathFromUri(uri: Uri): String {
        // For DocumentFile tree URIs, it often looks like "content://com.android.externalstorage.documents/tree/primary%3ADCIM%2FCamera"
        // We can try to extract a more readable name, or just use the last segment.
        val documentFile = DocumentFile.fromTreeUri(this, uri)
        return documentFile?.name ?: uri.path ?: "Selected Folder"
    }

    /**
     * Enables or disables the Start and Stop evaluation buttons.
     * @param enable If true, enables Start; if false, disables both.
     */
    private fun setEvaluationButtonsEnabled(enable: Boolean) {
        startButton.isEnabled = enable
        // Stop button is generally enabled only when evaluation is active
        stopButton.isEnabled = false
    }

    /**
     * Requests necessary runtime permissions (READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE).
     */
    private fun requestPermissionsIfNecessary() {
        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        } else {
            // Permissions are already granted, potentially enable Start button if dataset is chosen
            if (selectedDatasetUri != null) {
                validateDatasetFolder(selectedDatasetUri!!)
            }
        }
    }

    /**
     * Checks if all required permissions have been granted by the user.
     * @return True if all permissions are granted, false otherwise.
     */
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    /**
     * Callback for the result of requesting permissions.
     */
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                Toast.makeText(this, "Storage permissions granted.", Toast.LENGTH_SHORT).show()
                // If permissions are granted, re-validate dataset to enable Start button
                if (selectedDatasetUri != null) {
                    validateDatasetFolder(selectedDatasetUri!!)
                }
            } else {
                Toast.makeText(this, "Storage permissions not granted. App functionality may be limited.", Toast.LENGTH_LONG).show()
                // Optionally: finish() the app or disable critical features
                setEvaluationButtonsEnabled(false)
            }
        }
    }
}
