document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const questionInput = document.getElementById('question-input');
    const sendButton = document.getElementById('send-button');
    const uploadForm = document.getElementById('upload-form');
    const uploadStatus = document.getElementById('upload-status');
    const loadingIndicator = document.getElementById('loading-indicator');
    const uploadLoading = document.getElementById('upload-loading');
    
    // Object to store active processing jobs
    const activeJobs = {};
    
    // Send question
    function sendQuestion() {
        const question = questionInput.value.trim();
        if (!question) return;
        
        // Add user message to chat
        addMessage('User', question, 'user-message');
        
        // Clear input
        questionInput.value = '';
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        
        // Send question to API with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
        
        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question }),
            signal: controller.signal
        })
        .then(response => {
            clearTimeout(timeoutId);
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.detail || 'Server error occurred');
                }).catch(error => {
                    throw new Error(`Network error: ${response.status} ${response.statusText}`);
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
            
            // Extract source documents
            let sourcesHtml = '';
            if (data.source_documents && data.source_documents.length > 0) {
                sourcesHtml = '<div class="sources"><strong>Sources:</strong><ul>';
                data.source_documents.forEach(doc => {
                    sourcesHtml += `<li>${doc.metadata.source || 'Unknown source'}</li>`;
                });
                sourcesHtml += '</ul></div>';
            }
            
            // Add assistant message to chat
            const messageContent = `<div>${data.answer}</div>${sourcesHtml}`;
            addMessage('Assistant', messageContent, 'assistant-message', false);
        })
        .catch(error => {
            console.error('Error:', error);
            loadingIndicator.style.display = 'none';
            
            let errorMessage = 'Sorry, there was an error processing your question. Please try again.';
            if (error.name === 'AbortError') {
                errorMessage = 'The request took too long to complete. Please try a simpler question or try again later.';
            } else if (error.message) {
                errorMessage = `Error: ${error.message}`;
            }
            
            addMessage('Assistant', errorMessage, 'assistant-message', false);
        });
    }
    
    // Add message to chat
    function addMessage(sender, content, className, escapeHTML = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${className}`;
        
        const headerDiv = document.createElement('div');
        headerDiv.className = 'message-header';
        headerDiv.textContent = `${sender}:`;
        
        const contentDiv = document.createElement('div');
        if (escapeHTML) {
            contentDiv.textContent = content;
        } else {
            contentDiv.innerHTML = content;
        }
        
        messageDiv.appendChild(headerDiv);
        messageDiv.appendChild(contentDiv);
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Check processing status for a job
    function checkProcessingStatus(jobId) {
        if (!jobId || !activeJobs[jobId]) return;
        
        fetch(`/process-status/${jobId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error checking status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log(`Status for job ${jobId}:`, data);
            
            // Update status in UI
            if (data.status === 'complete') {
                // Processing completed successfully
                showUploadStatus(`Document processing complete: ${data.message}`, 'alert-success');
                
                // Remove from active jobs
                delete activeJobs[jobId];
                
                // Show success message in chat
                const chatTab = document.getElementById('chat-tab');
                addMessage('System', 'Your document has been successfully processed and is now available for questions.', 'assistant-message');
                chatTab.click();
                
            } else if (data.status === 'error') {
                // Processing failed
                showUploadStatus(`Error processing document: ${data.message}`, 'alert-danger');
                
                // Remove from active jobs
                delete activeJobs[jobId];
                
            } else if (data.status === 'processing' || data.status === 'pending') {
                // Still processing, update status and check again in a few seconds
                showUploadStatus(`Document processing in progress: ${data.message}`, 'alert-info');
                
                // Check again in 3 seconds
                setTimeout(() => checkProcessingStatus(jobId), 3000);
            }
        })
        .catch(error => {
            console.error('Error checking processing status:', error);
            
            // Try again in 5 seconds
            setTimeout(() => checkProcessingStatus(jobId), 5000);
        });
    }
    
    // Upload PDF
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('pdf-file');
        const file = fileInput.files[0];
        
        if (!file) {
            showUploadStatus('Please select a PDF file.', 'alert-danger');
            return;
        }
        
        // Check file type
        if (file.type !== 'application/pdf') {
            showUploadStatus('Please select a valid PDF file.', 'alert-danger');
            return;
        }
        
        // Check file size (limit to 10MB)
        const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
        if (file.size > MAX_FILE_SIZE) {
            showUploadStatus('File size exceeds the 10MB limit. Please select a smaller file.', 'alert-danger');
            return;
        }
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Show loading indicator
        uploadLoading.style.display = 'block';
        uploadStatus.style.display = 'none';
        
        // Send file to API with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout for uploads
        
        fetch('/upload-pdf', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        })
        .then(response => {
            clearTimeout(timeoutId);
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.detail || 'Server error occurred');
                }).catch(error => {
                    if (error instanceof SyntaxError) {
                        throw new Error(`Server error: ${response.status} ${response.statusText}`);
                    }
                    throw error;
                });
            }
            return response.json();
        })
        .then(data => {
            uploadLoading.style.display = 'none';
            showUploadStatus(`${data.message} Processing may take a few minutes for large documents.`, 'alert-info');
            uploadForm.reset();
            
            // Add to active jobs
            if (data.job_id) {
                activeJobs[data.job_id] = {
                    filename: file.name,
                    startTime: new Date()
                };
                
                // Start checking processing status
                setTimeout(() => checkProcessingStatus(data.job_id), 2000);
            }
            
            // Add system message to chat about the new document
            addMessage('System', `New document "${file.name}" has been uploaded and is being processed. It will be available for questions shortly.`, 'assistant-message');
        })
        .catch(error => {
            console.error('Error:', error);
            uploadLoading.style.display = 'none';
            
            let errorMessage = 'Error uploading PDF. Please try again.';
            if (error.name === 'AbortError') {
                errorMessage = 'The upload took too long to complete. Please try a smaller file or try again later.';
            } else if (error.message) {
                errorMessage = `Upload error: ${error.message}`;
            }
            
            showUploadStatus(errorMessage, 'alert-danger');
        });
    });
    
    // Show upload status
    function showUploadStatus(message, className) {
        uploadStatus.className = `alert ${className}`;
        uploadStatus.textContent = message;
        uploadStatus.style.display = 'block';
    }
    
    // Check for existing documents on page load
    function checkExistingDocuments() {
        fetch('/list-documents')
        .then(response => response.json())
        .then(data => {
            if (data.documents && data.documents.length > 0) {
                // We have some documents, add a message to chat
                const docNames = data.documents.map(doc => doc.filename).join(', ');
                addMessage('System', `I have ${data.documents.length} document(s) ready for questions: ${docNames}`, 'assistant-message');
                
                // Check for any processing documents
                data.documents.forEach(doc => {
                    if (doc.processing_status === 'processing' || doc.processing_status === 'pending') {
                        activeJobs[doc.job_id] = {
                            filename: doc.filename,
                            startTime: new Date(doc.upload_time * 1000)
                        };
                        
                        // Start checking status
                        checkProcessingStatus(doc.job_id);
                    }
                });
            }
        })
        .catch(error => {
            console.error('Error checking existing documents:', error);
        });
    }
    
    // Event listeners
    sendButton.addEventListener('click', sendQuestion);
    
    questionInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendQuestion();
        }
    });
    
    // Check for existing documents on page load
    checkExistingDocuments();
}); 