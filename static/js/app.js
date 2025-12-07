/**
 * DermaLens Frontend Logic
 */

// Global state
let selectedFile = null;
const API_BASE_URL = window.location.origin;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const infoContent = document.getElementById('infoContent');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const clearBtn = document.getElementById('clearBtn');

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
analyzeBtn.addEventListener('click', handleAnalyze);
newAnalysisBtn.addEventListener('click', handleNewAnalysis);
clearBtn.addEventListener('click', handleClearImage);

/**
 * Drag and drop handlers
 */
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('border-purple-500', 'bg-purple-50');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('border-purple-500', 'bg-purple-50');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('border-purple-500', 'bg-purple-50');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect({ target: { files } });
    }
}

/**
 * Handle file selection
 */
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length === 0) return;

    const file = files[0];

    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file (JPEG or PNG)');
        return;
    }

    // Validate file size (10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showError(`File is too large. Maximum size is 10MB. Your file is ${(file.size / (1024 * 1024)).toFixed(2)}MB`);
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        previewSection.classList.remove('hidden');
        analyzeBtn.disabled = false;
        analyzeBtn.classList.remove('disabled:from-gray-300', 'disabled:to-gray-300', 'disabled:cursor-not-allowed');
        hideError();
    };
    reader.readAsDataURL(file);
}

/**
 * Clear selected image
 */
function handleClearImage() {
    selectedFile = null;
    fileInput.value = '';
    previewSection.classList.add('hidden');
    preview.src = '';
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('disabled:from-gray-300', 'disabled:to-gray-300', 'disabled:cursor-not-allowed');
    resultsSection.classList.add('hidden');
}

/**
 * Handle image analysis
 */
async function handleAnalyze() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }

    // Disable analyze button and show loading
    analyzeBtn.disabled = true;
    loadingIndicator.classList.remove('hidden');
    hideError();
    resultsSection.classList.add('hidden');

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Send prediction request
        const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });

        const data = response.data;
        displayResults(data);

    } catch (error) {
        console.error('Prediction error:', error);
        if (error.response?.status === 503) {
            showError('Model is not loaded yet. Please ensure the model has been trained and placed in the checkpoints folder.');
        } else if (error.response?.data?.detail) {
            showError(error.response.data.detail);
        } else {
            showError('Error analyzing image. Please try again.');
        }
    } finally {
        analyzeBtn.disabled = false;
        loadingIndicator.classList.add('hidden');
    }
}

/**
 * Display prediction results
 */
function displayResults(data) {
    const predictedClass = data.predicted_class;
    const confidence = (data.confidence * 100).toFixed(1);
    const allPredictions = data.all_predictions;
    const classInfo = data.class_info;

    // Update prediction section
    document.getElementById('predictionClass').textContent = predictedClass;
    document.getElementById('confidenceScore').textContent = `${confidence}%`;
    document.getElementById('confidenceBar').style.width = `${confidence}%`;

    // Update all predictions
    const allPredictionsDiv = document.getElementById('allPredictions');
    allPredictionsDiv.innerHTML = '';

    // Sort predictions by confidence
    const sortedPredictions = Object.entries(allPredictions)
        .sort(([, a], [, b]) => b - a);

    sortedPredictions.forEach(([className, confScore]) => {
        const confPercentage = (confScore * 100).toFixed(1);
        const barWidth = (confScore * 100);

        const html = `
            <div>
                <div class="flex justify-between mb-1">
                    <span class="text-sm font-medium text-gray-700">${className}</span>
                    <span class="text-sm font-bold text-gray-600">${confPercentage}%</span>
                </div>
                <div class="w-full bg-gray-200 rounded-full h-2">
                    <div class="bg-gradient-to-r from-purple-400 to-pink-400 h-2 rounded-full transition-all" style="width: ${barWidth}%"></div>
                </div>
            </div>
        `;
        allPredictionsDiv.innerHTML += html;
    });

    // Update disease information
    document.getElementById('diseaseTitle').textContent = `${predictedClass} - ${classInfo.severity}`;
    document.getElementById('diseaseDescription').textContent = classInfo.description;
    document.getElementById('severityLevel').textContent = classInfo.severity;
    document.getElementById('recommendation').textContent = classInfo.recommendation;

    // Update severity color
    const severityElement = document.getElementById('severityLevel');
    severityElement.classList.remove('severity-high', 'severity-medium', 'severity-low', 'text-gray-700');

    if (classInfo.severity === 'High') {
        severityElement.classList.add('severity-high');
    } else if (classInfo.severity === 'Medium' || classInfo.severity === 'Low-Medium') {
        severityElement.classList.add('severity-medium');
    } else if (classInfo.severity === 'Low') {
        severityElement.classList.add('severity-low');
    } else {
        severityElement.classList.add('text-gray-700');
    }

    // Show results section
    resultsSection.classList.remove('hidden');

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }, 100);
}

/**
 * Handle new analysis
 */
function handleNewAnalysis() {
    handleClearImage();
    dropZone.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Show error message
 */
function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    errorSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
}

/**
 * Hide error message
 */
function hideError() {
    errorSection.classList.add('hidden');
}

// Setup error dismiss button
document.getElementById('errorDismissBtn').addEventListener('click', hideError);

/**
 * Initialize app
 */
async function initializeApp() {
    try {
        // Check API health
        const healthResponse = await axios.get(`${API_BASE_URL}/health`);
        console.log('API Health:', healthResponse.data);

        if (!healthResponse.data.model_loaded) {
            console.warn('Model not loaded. Please train the model first.');
        }

        // Load class information
        const classesResponse = await axios.get(`${API_BASE_URL}/classes`);
        console.log('Available classes:', classesResponse.data);

    } catch (error) {
        console.error('Failed to initialize app:', error);
        console.warn('Make sure the backend API is running on port 8000');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initializeApp);
