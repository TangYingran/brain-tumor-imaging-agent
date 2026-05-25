const API_BASE = '';

const App = {
    uploadedFiles: {
        t1: null,
        flair: null,
        t1ce: null,
        t2: null
    },
    currentSlice: 0,
    totalSlices: 0,
    viewMode: 'overlay',
    alpha: 0.4,
    chatHistory: [],
    isUploading: false,
    isProcessing: false,

    init() {
        this.setupFileUploads();
        this.setupEventListeners();
        this.checkStatus();
    },

    setupFileUploads() {
        const modalities = ['t1', 'flair', 't1ce', 't2'];

        modalities.forEach(mod => {
            const input = document.getElementById(`file${mod.charAt(0).toUpperCase() + mod.slice(1)}`);
            const zone = document.getElementById(`drop${mod.charAt(0).toUpperCase() + mod.slice(1)}`);

            input.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleFileSelect(mod, e.target.files[0]);
                }
            });

            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('dragover');
            });

            zone.addEventListener('dragleave', () => {
                zone.classList.remove('dragover');
            });

            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
                if (e.dataTransfer.files.length > 0) {
                    this.handleFileSelect(mod, e.dataTransfer.files[0]);
                }
            });
        });
    },

    handleFileSelect(mod, file) {
        const zone = document.getElementById(`drop${mod.charAt(0).toUpperCase() + mod.slice(1)}`);
        this.uploadedFiles[mod] = file;
        zone.classList.add('uploaded');
        this.updateUploadButton();
    },

    updateUploadButton() {
        const allUploaded = Object.values(this.uploadedFiles).every(f => f !== null);
        document.getElementById('uploadBtn').disabled = !allUploaded;
    },

    setupEventListeners() {
        document.getElementById('uploadBtn').addEventListener('click', () => this.uploadModalities());

        document.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.viewMode = btn.dataset.mode;
                this.loadSlice(this.currentSlice);
            });
        });

        document.getElementById('sliceSlider').addEventListener('input', (e) => {
            this.currentSlice = parseInt(e.target.value);
            document.getElementById('currentSlice').textContent = this.currentSlice;
            this.loadSlice(this.currentSlice);
        });

        document.getElementById('prevSlice').addEventListener('click', () => {
            if (this.currentSlice > 0) {
                this.currentSlice--;
                this.updateSliceSlider();
                this.loadSlice(this.currentSlice);
            }
        });

        document.getElementById('nextSlice').addEventListener('click', () => {
            if (this.currentSlice < this.totalSlices - 1) {
                this.currentSlice++;
                this.updateSliceSlider();
                this.loadSlice(this.currentSlice);
            }
        });

        document.getElementById('alphaSlider').addEventListener('input', (e) => {
            this.alpha = parseInt(e.target.value) / 100;
            document.getElementById('alphaValue').textContent = `${e.target.value}%`;
            this.loadSlice(this.currentSlice);
        });

        document.getElementById('sendBtn').addEventListener('click', () => this.sendMessage());

        document.getElementById('chatInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        document.getElementById('chatInput').addEventListener('input', (e) => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 100) + 'px';
        });

        document.getElementById('clearChat').addEventListener('click', () => this.clearChat());
        document.getElementById('downloadBtn').addEventListener('click', () => this.downloadSegmentation());

        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const question = btn.dataset.question;
                const chatInput = document.getElementById('chatInput');
                chatInput.value = question;
                chatInput.style.height = 'auto';
                chatInput.style.height = Math.min(chatInput.scrollHeight, 100) + 'px';
                chatInput.focus();
            });
        });
    },

    updateSliceSlider() {
        const slider = document.getElementById('sliceSlider');
        slider.value = this.currentSlice;
        document.getElementById('currentSlice').textContent = this.currentSlice;
    },

    async uploadModalities() {
        if (this.isUploading) return;

        const patientId = document.getElementById('patientIdInput').value;
        const messageEl = document.getElementById('uploadMessage');

        if (!this.uploadedFiles.t1 || !this.uploadedFiles.flair ||
            !this.uploadedFiles.t1ce || !this.uploadedFiles.t2) {
            messageEl.textContent = '请上传全部四个模态：T1 / FLAIR / T1CE / T2';
            messageEl.classList.remove('success');
            messageEl.classList.add('error');
            return;
        }

        this.isUploading = true;
        const btn = document.getElementById('uploadBtn');
        btn.disabled = true;
        btn.querySelector('.btn-text').textContent = '上传中...';
        messageEl.textContent = '正在上传模态数据...';
        messageEl.classList.remove('success', 'error');

        try {
            const formData = new FormData();
            formData.append('patient_id', patientId);
            formData.append('t1', this.uploadedFiles.t1);
            formData.append('flair', this.uploadedFiles.flair);
            formData.append('t1ce', this.uploadedFiles.t1ce);
            formData.append('t2', this.uploadedFiles.t2);

            const response = await fetch(`${API_BASE}/api/upload_modalities`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                messageEl.textContent = data.message;
                messageEl.classList.add('success');
                messageEl.classList.remove('error');
                document.getElementById('currentPatientId').textContent = data.patient_id;
                this.updatePatientStatus(true);
                this.showViewerPlaceholder(false);
            } else {
                messageEl.textContent = data.message;
                messageEl.classList.add('error');
                messageEl.classList.remove('success');
            }
        } catch (error) {
            messageEl.textContent = 'Upload failed: ' + error.message;
            messageEl.classList.add('error');
            messageEl.classList.remove('success');
        } finally {
            this.isUploading = false;
            btn.disabled = false;
            btn.querySelector('.btn-text').textContent = '上传模态';
        }
    },

    async checkStatus() {
        try {
            const response = await fetch(`${API_BASE}/api/status`);
            const data = await response.json();

            if (data.patient_id) {
                document.getElementById('currentPatientId').textContent = data.patient_id;
            }

            if (data.has_modalities) {
                this.updatePatientStatus(true);
            }

            if (data.num_slices) {
                this.totalSlices = data.num_slices;
                this.updateSliceControl(true);
            }
        } catch (error) {
            console.log('Status check failed:', error);
        }
    },

    updatePatientStatus(online) {
        const statusIndicator = document.querySelector('.status-indicator');
        const statusText = document.querySelector('.status-text');

        if (online) {
            statusIndicator.classList.add('online');
            statusText.textContent = '已连接';
        } else {
            statusIndicator.classList.remove('online');
            statusText.textContent = '就绪';
        }
    },

    showViewerPlaceholder(show) {
        const placeholder = document.getElementById('viewerPlaceholder');
        const canvas = document.getElementById('sliceCanvas');

        if (show) {
            placeholder.classList.remove('hidden');
            canvas.classList.add('hidden');
        } else {
            placeholder.classList.add('hidden');
            canvas.classList.remove('hidden');
        }
    },

    updateSliceControl(show) {
        const control = document.getElementById('sliceControl');
        if (show) {
            control.classList.remove('hidden');
        } else {
            control.classList.add('hidden');
        }
    },

    async loadSlice(z) {
        if (this.totalSlices === 0) return;

        try {
            const response = await fetch(
                `${API_BASE}/api/slice?z=${z}&mode=${this.viewMode}&alpha=${this.alpha}`
            );
            const data = await response.json();

            if (data.success && data.image) {
                const canvas = document.getElementById('sliceCanvas');
                const ctx = canvas.getContext('2d');
                const img = new Image();

                img.onload = () => {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                };

                img.src = 'data:image/png;base64,' + data.image;
            }
        } catch (error) {
            console.error('Failed to load slice:', error);
        }
    },

    async sendMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();

        if (!message || this.isProcessing) return;

        this.isProcessing = true;
        const sendBtn = document.getElementById('sendBtn');
        sendBtn.disabled = true;

        this.addChatMessage('user', message);
        input.value = '';
        input.style.height = 'auto';

        const loadingMsg = this.addChatMessage('assistant', '分析中...');

        try {
            const response = await fetch(`${API_BASE}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });

            const data = await response.json();

            this.updateChatMessage(loadingMsg, data.success ? data.message : data.message);

            if (data.success) {
                if (data.num_slices) {
                    this.totalSlices = data.num_slices;
                    this.currentSlice = data.init_slice || 0;
                    document.getElementById('totalSlices').textContent = this.totalSlices;
                    document.getElementById('sliceCount').textContent = `${this.currentSlice} / ${this.totalSlices}`;
                    this.updateSliceControl(true);
                    this.showViewerPlaceholder(false);
                    this.updateSliceSlider();
                    this.loadSlice(this.currentSlice);
                }

                document.getElementById('downloadBtn').disabled = false;
            }
        } catch (error) {
            this.updateChatMessage(loadingMsg, '发送失败：' + error.message);
        } finally {
            this.isProcessing = false;
            sendBtn.disabled = false;
        }
    },

    addChatMessage(role, content) {
        const container = document.getElementById('chatContainer');

        if (container.querySelector('.chat-welcome')) {
            container.innerHTML = '';
        }

        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-message ${role}`;

        const initial = role === 'assistant' ? 'A' : 'U';
        const time = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });

        msgDiv.innerHTML = `
            <div class="message-avatar">${initial}</div>
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(content)}</div>
                <div class="message-time">${time}</div>
            </div>
        `;

        container.appendChild(msgDiv);
        container.scrollTop = container.scrollHeight;

        return msgDiv.querySelector('.message-text');
    },

    updateChatMessage(msgElement, content) {
        msgElement.innerHTML = this.escapeHtml(content);
        msgElement.parentElement.parentElement.querySelector('.message-time').textContent =
            new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        let html = div.innerHTML.replace(/\n/g, '<br>');
        html = this.parseMarkdown(html);
        return html;
    },

    parseMarkdown(text) {
        const codeBlock = /```([\s\S]*?)```/g;
        const inlineCode = /`([^`]+)`/g;
        const bold = /\*\*([^*]+)\*\*/g;
        const italic = /\*([^*]+)\*/g;

        let result = text;
        const codeBlocks = [];
        result = result.replace(codeBlock, (match, code) => {
            codeBlocks.push(`<pre><code>${code.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre>`);
            return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
        });

        result = result.replace(inlineCode, '<code>$1</code>');
        result = result.replace(bold, '<strong>$1</strong>');
        result = result.replace(italic, '<em>$1</em>');

        codeBlocks.forEach((block, i) => {
            result = result.replace(`__CODE_BLOCK_${i}__`, block);
        });

        return result;
    },

    clearChat() {
        const container = document.getElementById('chatContainer');
        container.innerHTML = `
            <div class="chat-welcome">
                <div class="welcome-icon">
                    <svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="32" cy="32" r="26" stroke-dasharray="8 4" opacity="0.4"/>
                        <path d="M32 12 C22 12 14 22 14 32 C14 40 18 48 24 52 C20 58 24 62 32 62 C40 62 44 58 40 52 C46 48 50 40 50 32 C50 22 42 12 32 12"/>
                        <path d="M22 22 Q32 28 42 22 M20 32 Q32 38 44 32 M22 42 Q32 48 42 42" opacity="0.5"/>
                    </svg>
                </div>
                <h3>欢迎使用 BratsAgent</h3>
                <p>我是您的脑肿瘤影像辅助诊断助手。上传MRI模态数据后，我可以帮您完成：</p>
                <div class="welcome-features">
                    <div class="feature-section">
                        <h4>🔬 核心功能</h4>
                        <ul>
                            <li>3D脑肿瘤分割分析</li>
                            <li>肿瘤体积精确测量</li>
                            <li>分割结果可视化展示</li>
                        </ul>
                    </div>
                    <div class="feature-section">
                        <h4>📊 分析解读</h4>
                        <ul>
                            <li>分割结果专业解读</li>
                            <li>肿瘤变化趋势分析</li>
                            <li>生成结构化诊断报告</li>
                        </ul>
                    </div>
                    <div class="feature-section">
                        <h4>💡 知识问答</h4>
                        <ul>
                            <li>ET/TC/WT标签含义解释</li>
                            <li>脑肿瘤常见治疗方案</li>
                            <li>相关医学知识查询</li>
                        </ul>
                    </div>
                </div>
                <div class="welcome-tips">
                    <p><strong>💡 快速开始：</strong></p>
                    <p>1. 在左侧上传四个MRI模态文件（T1、FLAIR、T1CE、T2）</p>
                    <p>2. 在下方输入框中输入指令，例如："请进行脑肿瘤三维分割"</p>
                </div>
            </div>
        `;

        fetch(`${API_BASE}/api/clear`, { method: 'POST' });

        document.getElementById('currentPatientId').textContent = '--';
        document.getElementById('sliceCount').textContent = '0 / 0';
        this.updatePatientStatus(false);
        this.updateSliceControl(false);
        this.showViewerPlaceholder(true);
        document.getElementById('downloadBtn').disabled = true;

        this.uploadedFiles = { t1: null, flair: null, t1ce: null, t2: null };
        document.querySelectorAll('.upload-zone').forEach(zone => {
            zone.classList.remove('uploaded');
        });
        this.updateUploadButton();
    },

    async downloadSegmentation() {
        window.open(`${API_BASE}/api/download/segmentation`, '_blank');
    }
};

document.addEventListener('DOMContentLoaded', () => {
    App.init();
});