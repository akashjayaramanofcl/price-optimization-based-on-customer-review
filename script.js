// Price Optimization Dashboard JavaScript

class PriceOptimizationApp {
    constructor() {
        this.loadingModal = null;
        this.initializeModal();
        this.initializeEventListeners();
    }

    initializeModal() {
        // Wait for Bootstrap to be fully loaded
        document.addEventListener('DOMContentLoaded', () => {
            this.setupModal();
        });
        
        // If DOM is already loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.setupModal();
            });
        } else {
            this.setupModal();
        }
    }

    setupModal() {
        try {
            const modalElement = document.getElementById('loadingModal');
            if (modalElement && typeof bootstrap !== 'undefined') {
                this.loadingModal = new bootstrap.Modal(modalElement, {
                    backdrop: 'static',
                    keyboard: false
                });
                console.log('Modal initialized successfully');
            } else {
                console.warn('Modal element not found or Bootstrap not loaded');
                // Fallback: use a simple loading indicator
                this.loadingModal = null;
            }
        } catch (error) {
            console.error('Error initializing modal:', error);
            this.loadingModal = null;
        }
    }

    initializeEventListeners() {
        // Upload button click handler
        const uploadBtn = document.getElementById('uploadBtn');
        if (uploadBtn) {
            uploadBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleFileUpload();
            });
        }

        // File input change handler
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    this.validateFile(file);
                }
            });
        }
    }

    validateFile(file) {
        const messageContainer = document.getElementById('messageContainer');
        if (messageContainer) {
            messageContainer.innerHTML = '';
        }

        // Check file type
        if (!file.name.toLowerCase().endsWith('.csv')) {
            this.showMessage('Please select a CSV file.', 'danger');
            return false;
        }

        // Check file size (16MB limit)
        if (file.size > 16 * 1024 * 1024) {
            this.showMessage('File size must be less than 16MB.', 'danger');
            return false;
        }

        this.showMessage('File selected successfully! Click "Optimize Prices" to proceed.', 'success');
        return true;
    }

    async handleFileUpload() {
        const fileInput = document.getElementById('fileInput');
        const file = fileInput ? fileInput.files[0] : null;

        if (!file) {
            this.showMessage('Please select a file first.', 'warning');
            return;
        }

        if (!this.validateFile(file)) {
            return;
        }

        // Show loading indicator
        this.showLoadingIndicator();
        this.showProgressBar();

        // Prepare form data
        const formData = new FormData();
        formData.append('file', file);

        try {
            console.log('Starting file upload...');
            
            // Upload file and process
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            console.log('Response received:', response.status);
            const result = await response.json();

            // Hide loading indicator
            this.hideLoadingIndicator();
            this.hideProgressBar();

            if (response.ok && result.success) {
                this.handleSuccessResponse(result);
            } else {
                this.handleErrorResponse(result.error || 'Unknown error occurred');
            }

        } catch (error) {
            console.error('Upload error:', error);
            this.hideLoadingIndicator();
            this.hideProgressBar();
            this.handleErrorResponse('Network error occurred. Please check your connection and try again.');
        }
    }

    showLoadingIndicator() {
        // Try to use Bootstrap modal first
        if (this.loadingModal) {
            try {
                this.loadingModal.show();
                return;
            } catch (error) {
                console.warn('Error showing modal, using fallback:', error);
            }
        }

        // Fallback: Show inline loading indicator
        this.showInlineLoader();
    }

    hideLoadingIndicator() {
        // Try to hide Bootstrap modal first
        if (this.loadingModal) {
            try {
                this.loadingModal.hide();
                return;
            } catch (error) {
                console.warn('Error hiding modal, using fallback:', error);
            }
        }

        // Fallback: Hide inline loading indicator
        this.hideInlineLoader();
    }

    showInlineLoader() {
        const uploadBtn = document.getElementById('uploadBtn');
        if (uploadBtn) {
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = `
                <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                Processing...
            `;
        }

        // Show overlay
        const overlay = document.createElement('div');
        overlay.id = 'loadingOverlay';
        overlay.innerHTML = `
            <div class="d-flex justify-content-center align-items-center h-100">
                <div class="text-center">
                    <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <h5 class="mt-3">Processing Your Data</h5>
                    <p class="text-muted">Please wait while we optimize your prices...</p>
                </div>
            </div>
        `;
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            z-index: 9999;
            display: flex;
            color: white;
        `;
        document.body.appendChild(overlay);
    }

    hideInlineLoader() {
        const uploadBtn = document.getElementById('uploadBtn');
        if (uploadBtn) {
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = `
                <i class="fas fa-rocket me-2"></i>
                Optimize Prices
            `;
        }

        // Remove overlay
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.remove();
        }
    }

    handleSuccessResponse(result) {
        console.log('Success response received:', result);
        
        // Hide any previous error messages
        const messageContainer = document.getElementById('messageContainer');
        if (messageContainer) {
            messageContainer.innerHTML = '';
        }

        // Display success message
        this.showMessage(result.message || 'Price optimization completed successfully!', 'success');

        // Show results section
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.style.display = 'block';
            resultsSection.classList.add('fade-in');
        }

        // Populate summary statistics
        if (result.summary) {
            this.populateSummaryStats(result.summary);
        }

        // Create visualizations
        if (result.plots) {
            this.createVisualizations(result.plots);
        }

        // Populate data table
        if (result.data) {
            this.populateDataTable(result.data);
        }

        // Scroll to results
        setTimeout(() => {
            if (resultsSection) {
                resultsSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        }, 500);
    }

    handleErrorResponse(errorMessage) {
        console.error('Error response:', errorMessage);
        this.showMessage(errorMessage, 'danger');
        
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.style.display = 'none';
        }
    }

    populateSummaryStats(summary) {
        const summaryContainer = document.getElementById('summaryStats');
        if (!summaryContainer) return;
        
        const summaryCards = [
            {
                title: 'Total Products',
                value: summary.total_products?.toLocaleString() || '0',
                icon: 'fas fa-boxes',
                colorClass: 'bg-primary-gradient',
                animation: 'slide-in-left'
            },
            {
                title: 'Avg Original Price',
                value: `$${summary.avg_original_price || '0.00'}`,
                icon: 'fas fa-tag',
                colorClass: 'bg-info-gradient',
                animation: 'fade-in'
            },
            {
                title: 'Avg Optimized Price',
                value: `$${summary.avg_optimized_price || '0.00'}`,
                icon: 'fas fa-chart-line',
                colorClass: 'bg-success-gradient',
                animation: 'slide-in-right'
            },
            {
                title: 'Avg Price Change',
                value: `${summary.avg_price_change >= 0 ? '+' : ''}$${summary.avg_price_change || '0.00'}`,
                icon: (summary.avg_price_change || 0) >= 0 ? 'fas fa-arrow-up' : 'fas fa-arrow-down',
                colorClass: (summary.avg_price_change || 0) >= 0 ? 'bg-success-gradient' : 'bg-danger-gradient',
                animation: 'fade-in'
            },
            {
                title: 'Revenue Change',
                value: `${summary.total_revenue_change >= 0 ? '+' : ''}$${summary.total_revenue_change || '0.00'}`,
                icon: 'fas fa-dollar-sign',
                colorClass: (summary.total_revenue_change || 0) >= 0 ? 'bg-success-gradient' : 'bg-danger-gradient',
                animation: 'slide-in-left'
            },
            {
                title: 'Positive Reviews',
                value: summary.positive_reviews?.toLocaleString() || '0',
                icon: 'fas fa-smile',
                colorClass: 'bg-success-gradient',
                animation: 'slide-in-right'
            }
        ];

        summaryContainer.innerHTML = summaryCards.map((card, index) => `
            <div class="col-lg-2 col-md-4 col-sm-6 mb-3">
                <div class="card summary-card ${card.colorClass} ${card.animation}" style="animation-delay: ${index * 0.1}s">
                    <div class="card-body text-center">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <div>
                                <h6 class="card-title mb-1 text-white-50">${card.title}</h6>
                                <div class="display-6 mb-0">${card.value}</div>
                            </div>
                            <i class="${card.icon}"></i>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    createVisualizations(plots) {
        const chartIds = [
            { id: 'sentimentChart', plot: 'sentiment_dist' },
            { id: 'priceChart', plot: 'price_dist' },
            { id: 'priceChangeChart', plot: 'price_change' },
            { id: 'sentimentPriceChart', plot: 'sentiment_price' }
        ];

        chartIds.forEach(({ id, plot }) => {
            try {
                if (plots[plot]) {
                    const plotData = JSON.parse(plots[plot]);
                    const element = document.getElementById(id);
                    
                    if (element && plotData.data && plotData.layout) {
                        Plotly.newPlot(id, plotData.data, plotData.layout, {
                            responsive: true,
                            displayModeBar: false
                        });
                    }
                }
            } catch (error) {
                console.error(`Error creating ${id}:`, error);
            }
        });
    }

    populateDataTable(data) {
        const tableBody = document.querySelector('#resultsTable tbody');
        if (!tableBody || !Array.isArray(data)) return;
        
        tableBody.innerHTML = data.map(row => `
            <tr>
                <td>
                    <strong>${this.truncateText(row.product_name || 'Unknown', 30)}</strong>
                </td>
                <td>
                    <span class="badge bg-secondary">$${(row.price || 0).toFixed(2)}</span>
                </td>
                <td>
                    <span class="badge bg-primary">$${(row.optimized_price || 0).toFixed(2)}</span>
                </td>
                <td>
                    <span class="sentiment-${row.sentiment || 'neutral'}">${this.capitalizeFirst(row.sentiment || 'neutral')}</span>
                </td>
                <td>
                    <span class="${this.getPriceChangeClass(row.price_change || 0)}">
                        ${(row.price_change || 0) >= 0 ? '+' : ''}$${(row.price_change || 0).toFixed(2)}
                    </span>
                </td>
                <td>
                    <span class="${this.getPriceChangeClass(row.price_change_percent || 0)}">
                        ${(row.price_change_percent || 0) >= 0 ? '+' : ''}${(row.price_change_percent || 0).toFixed(2)}%
                    </span>
                </td>
            </tr>
        `).join('');
    }

    showMessage(message, type) {
        const messageContainer = document.getElementById('messageContainer');
        if (!messageContainer) return;
        
        const alertClass = `alert-${type}`;
        const iconClass = this.getAlertIcon(type);
        
        messageContainer.innerHTML = `
            <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
                <i class="${iconClass} me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
    }

    showProgressBar() {
        const progressSection = document.getElementById('progressSection');
        if (!progressSection) return;
        
        const progressBar = progressSection.querySelector('.progress-bar');
        if (!progressBar) return;
        
        progressSection.style.display = 'block';
        
        // Animate progress bar
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 95) progress = 95;
            
            progressBar.style.width = `${progress}%`;
            
            if (progress >= 95) {
                clearInterval(interval);
            }
        }, 200);
        
        // Store interval ID for cleanup
        this.progressInterval = interval;
    }

    hideProgressBar() {
        const progressSection = document.getElementById('progressSection');
        if (!progressSection) return;
        
        const progressBar = progressSection.querySelector('.progress-bar');
        if (!progressBar) return;
        
        // Clear interval if exists
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        
        // Complete the progress bar
        progressBar.style.width = '100%';
        
        setTimeout(() => {
            progressSection.style.display = 'none';
            progressBar.style.width = '0%';
        }, 500);
    }

    // Utility functions
    truncateText(text, maxLength) {
        if (!text || text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    capitalizeFirst(str) {
        if (!str) return '';
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    getPriceChangeClass(value) {
        if (value > 0) return 'price-increase';
        if (value < 0) return 'price-decrease';
        return 'price-neutral';
    }

    getAlertIcon(type) {
        const icons = {
            'success': 'fas fa-check-circle',
            'danger': 'fas fa-exclamation-triangle',
            'warning': 'fas fa-exclamation-circle',
            'info': 'fas fa-info-circle'
        };
        return icons[type] || 'fas fa-info-circle';
    }
}

// Initialize the application when DOM is loaded
let app;

function initializeApp() {
    try {
        app = new PriceOptimizationApp();
        console.log('Price Optimization App initialized successfully');
    } catch (error) {
        console.error('Error initializing app:', error);
    }
}

// Multiple ways to ensure initialization
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// Add window resize handler for responsive charts
window.addEventListener('resize', () => {
    const charts = ['sentimentChart', 'priceChart', 'priceChangeChart', 'sentimentPriceChart'];
    charts.forEach(chartId => {
        try {
            const chartElement = document.getElementById(chartId);
            if (chartElement && chartElement.data) {
                Plotly.Plots.resize(chartId);
            }
        } catch (error) {
            console.warn(`Error resizing chart ${chartId}:`, error);
        }
    });
});

// Utility functions
const Utils = {
    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    },

    formatPercentage(value, decimals = 2) {
        return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`;
    },

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};
