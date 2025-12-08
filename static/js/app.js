// Disease information database
const diseaseInfo = {
    'demodicosis': {
        description: 'Caused by Demodex mites that live in hair follicles. Can lead to hair loss, redness, and scaling. Common in puppies and immunocompromised dogs.',
        severity: 'medium',
        recommendations: [
            'Schedule a vet appointment for skin scraping diagnosis',
            'Avoid stressful situations that can weaken immune system',
            'Maintain clean bedding and living environment',
            'Consider immune system support supplements (vet-approved)',
            'Do not attempt to treat at home without veterinary guidance'
        ]
    },
    'Dermatitis': {
        description: 'Skin inflammation caused by direct contact with irritants or allergens such as certain plants, chemicals, or materials.',
        severity: 'low',
        recommendations: [
            'Identify and remove the source of irritation',
            'Rinse the affected area with cool water',
            'Avoid using harsh chemicals or cleaners around your pet',
            'Consider switching to hypoallergenic bedding or bowls',
            'Consult your vet if symptoms persist or worsen'
        ]
    },
    'Fungal_infections': {
        description: 'Fungal skin infections (like ringworm) that can cause circular patches of hair loss, scaly skin, and may be contagious to other pets and humans.',
        severity: 'medium',
        recommendations: [
            'Isolate affected pet from other animals and children',
            'Visit vet for fungal culture and treatment plan',
            'Clean and disinfect all bedding, toys, and living areas',
            'Wash hands after handling affected pet',
            'Complete full course of antifungal medication as prescribed'
        ]
    },
    'Healthy': {
        description: 'No signs of skin disease detected. Your dog\'s skin appears healthy with normal coloration and texture.',
        severity: 'low',
        recommendations: [
            'Continue regular grooming routine',
            'Maintain balanced diet for skin health',
            'Regular vet check-ups for preventive care',
            'Monitor for any changes in skin condition',
            'Provide adequate hydration and exercise'
        ]
    },
    'Hypersensitivity': {
        description: 'Allergic reaction causing skin irritation, itching, redness, and inflammation. Can be triggered by food, environmental factors, or flea bites.',
        severity: 'medium',
        recommendations: [
            'Identify and eliminate potential allergens',
            'Consider hypoallergenic diet trial',
            'Use flea prevention products year-round',
            'Keep indoor environment clean and dust-free',
            'Consult vet about antihistamines or allergy testing'
        ]
    },
    'ringworm': {
        description: 'Highly contagious fungal infection causing circular, scaly patches of hair loss. Despite the name, it is not caused by worms.',
        severity: 'high',
        recommendations: [
            'Seek immediate veterinary treatment',
            'Quarantine infected pet from other animals and people',
            'Thoroughly disinfect all surfaces and items',
            'Wear gloves when handling pet or cleaning',
            'All household pets should be examined by vet'
        ]
    }
};

// DOM Elements
const fileInput = document.getElementById('fileInput');
const heroSection = document.getElementById('heroSection');
const uploadSection = document.getElementById('uploadSection');
const dropArea = document.getElementById('dropArea');
const uploadPrompt = document.getElementById('uploadPrompt');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const analyzeButtonContainer = document.getElementById('analyzeButtonContainer');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const resultImage = document.getElementById('resultImage');

let currentFile = null;

// File Input
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drop Area Click
dropArea.addEventListener('click', () => {
    if (!currentFile) {
        fileInput.click();
    }
});

// Drag and Drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => {
        dropArea.classList.add('drag-over');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => {
        dropArea.classList.remove('drag-over');
    }, false);
});

dropArea.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFile(files[0]);
}, false);

// Handle File
function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
    }

    currentFile = file;
    const reader = new FileReader();

    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadPrompt.classList.add('hidden');
        preview.classList.remove('hidden');
        analyzeButtonContainer.classList.remove('hidden');
    };

    reader.readAsDataURL(file);
}

function resetUpload() {
    currentFile = null;
    fileInput.value = '';
    previewImage.src = '';
    preview.classList.add('hidden');
    analyzeButtonContainer.classList.add('hidden');
    uploadPrompt.classList.remove('hidden');
}

function resetToUpload() {
    resultsSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    resetUpload();
}

// Analyze Button
analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Hide upload section, show loading
    uploadSection.classList.add('hidden');
    loading.classList.remove('hidden');

    try {
        const formData = new FormData();
        formData.append('file', currentFile);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        alert('Analysis failed. Please try again.');
        resetToUpload();
    } finally {
        loading.classList.add('hidden');
    }
});

// New Analysis Button
newAnalysisBtn.addEventListener('click', () => {
    resetToUpload();
});

// Display Results
function displayResults(result) {
    const predictedClass = result.predicted_class;
    const confidence = Math.round(result.confidence * 100);

    // Use class_info from API response, fallback to local data if not available
    const classInfo = result.class_info || diseaseInfo[predictedClass] || diseaseInfo['Dermatitis'];

    // Set result image
    resultImage.src = previewImage.src;

    // Set diagnosis name
    document.getElementById('diagnosisName').textContent = formatDiseaseName(predictedClass);

    // Determine severity level from API response
    let severityLevel = 'low';
    if (classInfo.severity) {
        const severityStr = classInfo.severity.toLowerCase();
        if (severityStr.includes('high')) {
            severityLevel = 'high';
        } else if (severityStr.includes('medium') || severityStr.includes('low-medium')) {
            severityLevel = 'medium';
        } else if (severityStr === 'none') {
            severityLevel = 'none';
        }
    }

    // Set severity badge with dynamic styling
    const severityBadge = document.getElementById('severityBadge');
    const severityText = document.getElementById('severityText');
    const severityIcon = severityBadge.querySelector('svg');

    // Update severity text
    severityText.textContent = classInfo.severity || severityLevel;

    // Apply severity-based styling
    if (severityLevel === 'high') {
        severityBadge.className = 'flex items-center gap-2 bg-red-100 rounded-full px-4 py-2';
        severityIcon.setAttribute('class', 'w-5 h-5 text-red-600');
        severityText.className = 'text-red-600 text-base uppercase font-medium';
    } else if (severityLevel === 'medium') {
        severityBadge.className = 'flex items-center gap-2 bg-yellow-100 rounded-full px-4 py-2';
        severityIcon.setAttribute('class', 'w-5 h-5 text-yellow-600');
        severityText.className = 'text-yellow-600 text-base uppercase font-medium';
    } else if (severityLevel === 'none') {
        severityBadge.className = 'flex items-center gap-2 bg-green-100 rounded-full px-4 py-2';
        severityIcon.setAttribute('class', 'w-5 h-5 text-green-600');
        severityText.className = 'text-green-600 text-base uppercase font-medium';
    } else {
        severityBadge.className = 'flex items-center gap-2 bg-[#4A8F6F]/10 rounded-full px-4 py-2';
        severityIcon.setAttribute('class', 'w-5 h-5 text-[#4A8F6F]');
        severityText.className = 'text-[#4A8F6F] text-base uppercase font-medium';
    }

    // Set confidence with color coding based on level
    const confidenceTextEl = document.getElementById('confidenceText');
    confidenceTextEl.textContent = `${confidence}% confidence`;

    const confidenceBar = document.getElementById('confidenceBar');
    let confidenceColor = '';

    // Color code confidence bar and determine color for severity badge
    if (confidence >= 75) {
        confidenceBar.className = 'bg-green-600 h-3 rounded-full transition-all duration-500';
        confidenceColor = 'green';
    } else if (confidence >= 50) {
        confidenceBar.className = 'bg-yellow-500 h-3 rounded-full transition-all duration-500';
        confidenceColor = 'yellow';
    } else {
        confidenceBar.className = 'bg-red-500 h-3 rounded-full transition-all duration-500';
        confidenceColor = 'red';
    }

    // Override severity badge color to match confidence bar for better visual consistency
    if (severityLevel === 'high') {
        // Use red for high severity regardless of confidence
        severityBadge.className = 'flex items-center gap-2 bg-red-100 rounded-full px-4 py-2';
        severityIcon.class = 'w-5 h-5 text-red-600';
        severityText.className = 'text-red-600 text-base uppercase font-medium';
    } else if (severityLevel === 'medium') {
        // Use yellow for medium severity regardless of confidence
        severityBadge.className = 'flex items-center gap-2 bg-yellow-100 rounded-full px-4 py-2';
        severityIcon.class = 'w-5 h-5 text-yellow-600';
        severityText.className = 'text-yellow-600 text-base uppercase font-medium';
    } else if (severityLevel === 'none') {
        // Use green for none/healthy
        severityBadge.className = 'flex items-center gap-2 bg-green-100 rounded-full px-4 py-2';
        severityIcon.class = 'w-5 h-5 text-green-600';
        severityText.className = 'text-green-600 text-base uppercase font-medium';
    } else {
        // For low severity, use teal/green
        severityBadge.className = 'flex items-center gap-2 bg-[#4A8F6F]/10 rounded-full px-4 py-2';
        severityIcon.class = 'w-5 h-5 text-[#4A8F6F]';
        severityText.className = 'text-[#4A8F6F] text-base uppercase font-medium';
    }

    setTimeout(() => {
        confidenceBar.style.width = `${confidence}%`;
    }, 100);

    // Set description from API response
    const description = classInfo.description || 'No detailed description available.';
    document.getElementById('diseaseDescription').textContent = description;

    // Set recommendations from API response
    const recommendationsList = document.getElementById('recommendationsList');

    // Check if recommendations exist in API response
    let recommendations = [];
    if (classInfo.recommendation) {
        // Split recommendation by periods or newlines
        recommendations = classInfo.recommendation
            .split(/[.\n]\s*/)
            .filter(rec => rec.trim().length > 0)
            .map(rec => rec.trim());
    }

    // Fallback to local recommendations if API doesn't provide them
    if (recommendations.length === 0 && diseaseInfo[predictedClass]) {
        recommendations = diseaseInfo[predictedClass].recommendations;
    }

    // If still no recommendations, provide a default
    if (recommendations.length === 0) {
        recommendations = ['Consult with a licensed veterinarian for proper diagnosis and treatment'];
    }

    recommendationsList.innerHTML = recommendations.map(rec => `
        <li class="flex items-start gap-6">
            <div class="w-2 h-2 rounded-full bg-[#1A1A1A] flex-shrink-0 mt-3"></div>
            <span class="text-[#6B6B6B] text-base leading-[1.625] flex-1">${rec}</span>
        </li>
    `).join('');

    // Populate other predictions dropdown
    if (result.all_predictions) {
        console.log('All predictions:', result.all_predictions);

        const otherPredictionsContainer = document.getElementById('otherPredictions');
        const predictions = Object.entries(result.all_predictions)
            .map(([className, conf]) => ({
                name: className,
                confidence: Math.round(conf * 100)
            }))
            .sort((a, b) => b.confidence - a.confidence)
            .filter(pred => pred.name !== predictedClass); // Exclude the main prediction

        otherPredictionsContainer.innerHTML = predictions.map(pred => {
            let barColor = '';
            if (pred.confidence >= 75) {
                barColor = 'bg-green-600';
            } else if (pred.confidence >= 50) {
                barColor = 'bg-yellow-500';
            } else {
                barColor = 'bg-red-500';
            }

            return `
                <div class="flex items-center gap-4">
                    <div class="flex-1">
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-sm font-medium text-[#1A1A1A]">${formatDiseaseName(pred.name)}</span>
                            <span class="text-sm text-[#6B6B6B]">${pred.confidence}%</span>
                        </div>
                        <div class="bg-[#E8E4DF] rounded-full h-2">
                            <div class="${barColor} h-2 rounded-full transition-all duration-500" style="width: ${pred.confidence}%"></div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    // Scroll to results and show
    resultsSection.classList.remove('hidden');
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// Helper function to format disease names
function formatDiseaseName(name) {
    const nameMap = {
        'demodicosis': 'Demodicosis',
        'Demodicosis': 'Demodicosis',
        'Dermatitis': 'Contact Dermatitis',
        'dermatitis': 'Contact Dermatitis',
        'Fungal_infections': 'Fungal Infection',
        'Fungal Infections': 'Fungal Infection',
        'fungal_infections': 'Fungal Infection',
        'Healthy': 'Healthy Skin',
        'healthy': 'Healthy Skin',
        'Hypersensitivity': 'Hypersensitivity',
        'hypersensitivity': 'Hypersensitivity',
        'ringworm': 'Ringworm',
        'Ringworm': 'Ringworm'
    };
    return nameMap[name] || name;
}

// New Analysis Button
document.getElementById('newAnalysisBtn').addEventListener('click', () => {
    resultsSection.classList.add('hidden');
    uploadPrompt.classList.remove('hidden');
    imagePreview.classList.add('hidden');
    fileInput.value = '';
    document.getElementById('uploadSection').scrollIntoView({ behavior: 'smooth' });
});

// Toggle Other Possibilities Dropdown
document.getElementById('togglePossibilities').addEventListener('click', () => {
    const dropdown = document.getElementById('otherPredictions');
    const icon = document.getElementById('dropdownIcon');

    if (dropdown.classList.contains('hidden')) {
        dropdown.classList.remove('hidden');
        icon.style.transform = 'rotate(180deg)';
    } else {
        dropdown.classList.add('hidden');
        icon.style.transform = 'rotate(0deg)';
    }
});
