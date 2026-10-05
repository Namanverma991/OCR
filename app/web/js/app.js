/**
 * PaddleOCR Studio Frontend Interactive Application
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const processBtn = document.getElementById('process-btn');
    const sampleBtn = document.getElementById('sample-btn');
    const langSelect = document.getElementById('lang-select');
    const deskewToggle = document.getElementById('deskew-toggle');
    const contrastToggle = document.getElementById('contrast-toggle');
    const angleClsToggle = document.getElementById('angle-cls-toggle');

    const viewportWrapper = document.getElementById('viewport-wrapper');
    const canvasContainer = document.getElementById('canvas-container');
    const canvasPlaceholder = document.getElementById('canvas-placeholder');
    const sourceImage = document.getElementById('source-image');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const canvasTools = document.getElementById('canvas-tools');
    const boxTooltip = document.getElementById('box-tooltip');
    const showBoxesChk = document.getElementById('show-boxes-chk');

    // Metrics
    const metricTime = document.getElementById('metric-time');
    const metricLines = document.getElementById('metric-lines');
    const metricConf = document.getElementById('metric-conf');

    // Tab panes
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const outputTextArea = document.getElementById('output-text-area');
    const entitiesContainer = document.getElementById('entities-container');
    const tableRenderArea = document.getElementById('table-render-area');
    const jsonViewerPre = document.getElementById('json-viewer-pre');

    // Action buttons
    const copyTextBtn = document.getElementById('copy-text-btn');
    const downloadTxtBtn = document.getElementById('download-txt-btn');
    const copyJsonBtn = document.getElementById('copy-json-btn');
    const downloadJsonBtn = document.getElementById('download-json-btn');
    const downloadCsvBtn = document.getElementById('download-csv-btn');
    const downloadMdBtn = document.getElementById('download-md-btn');
    const extractEntitiesBtn = document.getElementById('extract-entities-btn');

    let currentFile = null;
    let ocrResults = null;
    let hoveredBoxIndex = -1;
    let scaleRatio = 1.0;

    // --- File Drag and Drop Handlers ---
    browseBtn.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('click', (e) => {
        if (e.target !== browseBtn) fileInput.click();
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFileSelection(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileSelection(e.target.files[0]);
    });

    function handleFileSelection(file) {
        currentFile = file;
        processBtn.disabled = false;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            sourceImage.src = e.target.result;
            sourceImage.onload = () => {
                canvasPlaceholder.style.display = 'none';
                canvasContainer.style.display = 'inline-block';
                canvasTools.style.display = 'flex';
                setupCanvas();
                showToast(`Loaded "${file.name}" ready for OCR`);
            };
        };
        reader.readAsDataURL(file);
    }

    // --- Tab Switching ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.getAttribute('data-tab');
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(target).classList.add('active');
        });
    });

    // --- Canvas & Box Drawing ---
    function setupCanvas() {
        const naturalW = sourceImage.naturalWidth;
        const naturalH = sourceImage.naturalHeight;
        
        overlayCanvas.width = naturalW;
        overlayCanvas.height = naturalH;
        
        drawBoxes();
    }

    function drawBoxes() {
        const ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

        if (!ocrResults || !showBoxesChk.checked) return;

        ocrResults.results.forEach((item, idx) => {
            const isHovered = idx === hoveredBoxIndex;
            const pts = item.box;
            if (!pts || pts.length < 4) return;

            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            for (let i = 1; i < pts.length; i++) {
                ctx.lineTo(pts[i][0], pts[i][1]);
            }
            ctx.closePath();

            // Confidence based coloring
            let strokeColor = '#06b6d4'; // Cyan
            let fillColor = 'rgba(6, 182, 212, 0.15)';
            if (item.confidence >= 0.90) {
                strokeColor = '#10b981'; // Emerald
                fillColor = isHovered ? 'rgba(16, 185, 129, 0.35)' : 'rgba(16, 185, 129, 0.15)';
            } else if (item.confidence < 0.75) {
                strokeColor = '#f43f5e'; // Rose
                fillColor = isHovered ? 'rgba(244, 63, 94, 0.35)' : 'rgba(244, 63, 94, 0.15)';
            }

            if (isHovered) {
                strokeColor = '#ffffff';
                ctx.lineWidth = 3;
            } else {
                ctx.lineWidth = 2;
            }

            ctx.strokeStyle = strokeColor;
            ctx.fillStyle = fillColor;
            ctx.fill();
            ctx.stroke();
        });
    }

    // --- Canvas Mouse Hover / Tooltip ---
    overlayCanvas.addEventListener('mousemove', (e) => {
        if (!ocrResults || !ocrResults.results) return;

        const rect = overlayCanvas.getBoundingClientRect();
        const scaleX = overlayCanvas.width / rect.width;
        const scaleY = overlayCanvas.height / rect.height;

        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top) * scaleY;

        let foundIdx = -1;
        for (let i = 0; i < ocrResults.results.length; i++) {
            const item = ocrResults.results[i];
            const pts = item.box;
            if (isPointInPolygon([mouseX, mouseY], pts)) {
                foundIdx = i;
                break;
            }
        }

        if (foundIdx !== -1) {
            hoveredBoxIndex = foundIdx;
            const item = ocrResults.results[foundIdx];
            
            // Show Tooltip
            boxTooltip.style.display = 'block';
            boxTooltip.style.left = `${e.clientX - viewportWrapper.getBoundingClientRect().left + 15}px`;
            boxTooltip.style.top = `${e.clientY - viewportWrapper.getBoundingClientRect().top + 15}px`;
            document.getElementById('tt-conf').textContent = `${(item.confidence * 100).toFixed(1)}% Conf`;
            document.getElementById('tt-angle').textContent = `${item.angle || 0}° Angle`;
            document.getElementById('tt-text').textContent = item.text;
        } else {
            hoveredBoxIndex = -1;
            boxTooltip.style.display = 'none';
        }

        drawBoxes();
    });

    overlayCanvas.addEventListener('mouseleave', () => {
        hoveredBoxIndex = -1;
        boxTooltip.style.display = 'none';
        drawBoxes();
    });

    showBoxesChk.addEventListener('change', drawBoxes);

    function isPointInPolygon(point, vs) {
        const x = point[0], y = point[1];
        let inside = false;
        for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
            const xi = vs[i][0], yi = vs[i][1];
            const xj = vs[j][0], yj = vs[j][1];
            const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    // --- Process OCR Request ---
    processBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        processBtn.disabled = true;
        processBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Neural Pipeline...';

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('lang', langSelect.value);
        formData.append('preprocess', deskewToggle.checked ? 'true' : 'false');
        formData.append('use_angle_cls', angleClsToggle.checked ? 'true' : 'false');

        try {
            const res = await fetch('/api/v1/ocr/image', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error(`Server returned status: ${res.status}`);
            
            ocrResults = await res.json();
            renderOCRResults(ocrResults);
            showToast('OCR extraction completed successfully!');
            
            // Also query structured & table endpoints in parallel
            fetchStructured(currentFile);
            fetchTable(currentFile);

        } catch (err) {
            console.error(err);
            showToast(`Error: ${err.message}`, true);
        } finally {
            processBtn.disabled = false;
            processBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run OCR Extraction';
        }
    });

    function renderOCRResults(data) {
        // Update metrics
        metricTime.textContent = `${data.processing_time_ms} ms`;
        metricLines.textContent = data.total_lines;
        
        let avgConf = 0;
        if (data.results && data.results.length > 0) {
            avgConf = data.results.reduce((acc, it) => acc + it.confidence, 0) / data.results.length;
        }
        metricConf.textContent = `${(avgConf * 100).toFixed(1)}%`;

        // Text Pane
        outputTextArea.value = data.full_text;

        // JSON Pane
        jsonViewerPre.querySelector('code').textContent = JSON.stringify(data, null, 2);

        // Render Canvas
        setupCanvas();
    }

    async function fetchStructured(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            const res = await fetch('/api/v1/ocr/structured', { method: 'POST', body: formData });
            if (res.ok) {
                const sdata = await res.json();
                renderEntities(sdata.entities);
            }
        } catch (e) {
            console.warn('Structured parse failed', e);
        }
    }

    function renderEntities(entities) {
        entitiesContainer.innerHTML = '';
        const keys = Object.keys(entities);
        if (keys.length === 0) {
            entitiesContainer.innerHTML = '<p class="empty-state-text">No entities extracted.</p>';
            return;
        }

        keys.forEach(k => {
            const val = entities[k];
            if (!val || (Array.isArray(val) && val.length === 0)) return;

            const row = document.createElement('div');
            row.className = 'entity-row';
            const displayVal = Array.isArray(val) ? val.join(', ') : val;
            row.innerHTML = `
                <span class="entity-key">${k.replace(/_/g, ' ')}</span>
                <span class="entity-val">${displayVal}</span>
            `;
            entitiesContainer.appendChild(row);
        });
    }

    async function fetchTable(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            const res = await fetch('/api/v1/ocr/table', { method: 'POST', body: formData });
            if (res.ok) {
                const tdata = await res.json();
                renderTableData(tdata);
            }
        } catch (e) {
            console.warn('Table extract failed', e);
        }
    }

    function renderTableData(tdata) {
        if (!tdata.tables || tdata.tables.length === 0 || tdata.tables[0].rows === 0) {
            tableRenderArea.innerHTML = '<p class="empty-state-text">No tables detected in document.</p>';
            return;
        }

        const table = tdata.tables[0];
        let html = '<table class="custom-table">';
        
        if (table.headers && table.headers.length > 0) {
            html += '<thead><tr>';
            table.headers.forEach(h => html += `<th>${h}</th>`);
            html += '</tr></thead>';
        }

        html += '<tbody>';
        table.matrix.slice(1).forEach(row => {
            html += '<tr>';
            row.forEach(cell => html += `<td>${cell}</td>`);
            html += '</tr>';
        });
        html += '</tbody></table>';

        tableRenderArea.innerHTML = html;
    }

    // --- Sample Invoice Generator ---
    sampleBtn.addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        canvas.width = 700;
        canvas.height = 900;
        const ctx = canvas.getContext('2d');

        // Draw Invoice Sample
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, 700, 900);

        ctx.fillStyle = '#1e293b';
        ctx.font = 'bold 28px sans-serif';
        ctx.fillText('ACME SOLUTIONS INC.', 50, 70);

        ctx.font = '14px sans-serif';
        ctx.fillStyle = '#64748b';
        ctx.fillText('104 Innovation Way, Tech Park, CA 94016', 50, 95);
        ctx.fillText('support@acmesolutions.ai | +1 (555) 019-2834', 50, 115);

        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(50, 135);
        ctx.lineTo(650, 135);
        ctx.stroke();

        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 18px sans-serif';
        ctx.fillText('TAX INVOICE', 50, 175);
        ctx.font = '14px sans-serif';
        ctx.fillText('Invoice Number: INV-2026-9941', 50, 205);
        ctx.fillText('Invoice Date: Oct 05, 2026', 50, 230);
        ctx.fillText('Tax ID: US-991204-TAX', 50, 255);

        // Table Header
        ctx.fillStyle = '#f1f5f9';
        ctx.fillRect(50, 300, 600, 35);
        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText('Description', 65, 323);
        ctx.fillText('Qty', 380, 323);
        ctx.fillText('Rate ($)', 460, 323);
        ctx.fillText('Amount ($)', 560, 323);

        // Table Rows
        const items = [
            ['PaddleOCR Neural Engine License', '1', '1200.00', '1200.00'],
            ['GPU Acceleration Pipeline Addon', '2', '450.00', '900.00'],
            ['High-Throughput Batch Processing', '1', '650.00', '650.00'],
            ['Cloud Multi-Region SLA Support', '1', '350.00', '350.00']
        ];

        ctx.font = '13px sans-serif';
        let y = 365;
        items.forEach((row, i) => {
            ctx.fillStyle = i % 2 === 0 ? '#ffffff' : '#f8fafc';
            ctx.fillRect(50, y - 20, 600, 30);
            ctx.fillStyle = '#334155';
            ctx.fillText(row[0], 65, y);
            ctx.fillText(row[1], 380, y);
            ctx.fillText(row[2], 460, y);
            ctx.fillText(row[3], 560, y);
            y += 35;
        });

        // Totals
        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 16px sans-serif';
        ctx.fillText('Total Amount Due: $3,100.00', 410, 560);

        canvas.toBlob((blob) => {
            const sampleFile = new File([blob], 'sample_invoice.png', { type: 'image/png' });
            handleFileSelection(sampleFile);
        });
    });

    // --- Clipboard & Downloads ---
    copyTextBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(outputTextArea.value);
        showToast('Text copied to clipboard!');
    });

    copyJsonBtn.addEventListener('click', () => {
        if (ocrResults) {
            navigator.clipboard.writeText(JSON.stringify(ocrResults, null, 2));
            showToast('JSON copied to clipboard!');
        }
    });

    downloadTxtBtn.addEventListener('click', () => {
        downloadFile('extracted_text.txt', outputTextArea.value, 'text/plain');
    });

    downloadJsonBtn.addEventListener('click', () => {
        if (ocrResults) {
            downloadFile('ocr_results.json', JSON.stringify(ocrResults, null, 2), 'application/json');
        }
    });

    function downloadFile(filename, content, mime) {
        const blob = new Blob([content], { type: mime });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }

    // --- Toast Notifications ---
    function showToast(msg, isError = false) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        if (isError) toast.style.borderColor = '#f43f5e';
        toast.textContent = msg;
        document.getElementById('toast-container').appendChild(toast);
        setTimeout(() => toast.remove(), 4000);
    }
});
