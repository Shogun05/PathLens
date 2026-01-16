import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

export interface AmenityPlacement {
    type: string;
    nodeId: string;
    latitude: number;
    longitude: number;
    description?: string;
}

export interface ReportData {
    location: string;
    baselineScore: number;
    optimizedScore: number;
    amenities: AmenityPlacement[];
    generatedAt: Date;
}

// Color palette matching PathLens branding
const COLORS = {
    primary: '#8fd6ff',
    dark: '#0f1c23',
    accent: '#10b981',
    text: '#333333',
    textLight: '#666666',
    border: '#e0e0e0',
    headerBg: '#1b2328',
    rowAlt: '#f8fafc',
};

// Amenity type colors for visual distinction
const AMENITY_COLORS: Record<string, string> = {
    bank: '#3b82f6',
    hospital: '#ef4444',
    school: '#f59e0b',
    pharmacy: '#8b5cf6',
    supermarket: '#10b981',
    bus_station: '#06b6d4',
    park: '#22c55e',
};

/**
 * Generate a professional PDF report for PathLens optimization results
 */
export async function generateOptimizationReport(
    mapContainer: HTMLElement | null,
    data: ReportData
): Promise<void> {
    const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4',
    });

    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const margin = 15;
    const contentWidth = pageWidth - margin * 2;

    let yPos = margin;

    // === PAGE 1: Header & Map ===

    // Header background
    pdf.setFillColor(27, 35, 40); // #1b2328
    pdf.rect(0, 0, pageWidth, 45, 'F');

    // Logo/Title
    pdf.setTextColor(143, 214, 255); // #8fd6ff
    pdf.setFontSize(24);
    pdf.setFont('helvetica', 'bold');
    pdf.text('PATHLENS', margin, 20);

    pdf.setFontSize(14);
    pdf.setTextColor(255, 255, 255);
    pdf.text('Optimization Report', margin, 30);

    // Metadata on right side
    pdf.setFontSize(10);
    pdf.setTextColor(200, 200, 200);
    const dateStr = data.generatedAt.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
    });
    pdf.text(dateStr, pageWidth - margin, 20, { align: 'right' });
    pdf.text(data.location || 'Urban Analysis', pageWidth - margin, 28, { align: 'right' });

    yPos = 55;

    // === Map Section ===
    if (mapContainer) {
        try {
            pdf.setFontSize(12);
            pdf.setTextColor(51, 51, 51);
            pdf.setFont('helvetica', 'bold');
            pdf.text('Optimized Amenity Placements', margin, yPos);
            yPos += 8;

            // Capture the map with improved options
            const canvas = await html2canvas(mapContainer, {
                useCORS: true,
                allowTaint: true,
                scale: 1.5, // Reduced scale for performance
                logging: true, // Enable for debugging
                backgroundColor: '#1a1a2e',
                foreignObjectRendering: false, // Disable for better tile capture
                removeContainer: true,
                imageTimeout: 15000, // Wait for tiles to load
                onclone: (clonedDoc) => {
                    // Ensure leaflet controls are visible in clone
                    const controls = clonedDoc.querySelectorAll('.leaflet-control');
                    controls.forEach((ctrl) => {
                        (ctrl as HTMLElement).style.display = 'none';
                    });
                },
            });

            if (canvas.width > 0 && canvas.height > 0) {
                const imgData = canvas.toDataURL('image/png');
                const imgWidth = contentWidth;
                const imgHeight = (canvas.height / canvas.width) * imgWidth;
                const maxImgHeight = 100; // Limit map height

                const finalHeight = Math.min(imgHeight, maxImgHeight);

                // Add border around map
                pdf.setDrawColor(200, 200, 200);
                pdf.setLineWidth(0.5);
                pdf.rect(margin - 1, yPos - 1, contentWidth + 2, finalHeight + 2);

                pdf.addImage(imgData, 'PNG', margin, yPos, imgWidth, finalHeight);
                yPos += finalHeight + 10;
            } else {
                throw new Error('Canvas has no dimensions');
            }
        } catch (error) {
            console.error('Failed to capture map:', error);
            // Draw a placeholder box with message
            pdf.setFillColor(240, 240, 240);
            pdf.setDrawColor(200, 200, 200);
            pdf.roundedRect(margin, yPos, contentWidth, 50, 3, 3, 'FD');
            pdf.setFontSize(10);
            pdf.setTextColor(100, 100, 100);
            pdf.text('Map preview not available', pageWidth / 2, yPos + 25, { align: 'center' });
            pdf.setFontSize(8);
            pdf.text('(View map in the application)', pageWidth / 2, yPos + 33, { align: 'center' });
            yPos += 60;
        }
    } else {
        // No map container - draw placeholder
        pdf.setFillColor(240, 240, 240);
        pdf.setDrawColor(200, 200, 200);
        pdf.roundedRect(margin, yPos, contentWidth, 50, 3, 3, 'FD');
        pdf.setFontSize(10);
        pdf.setTextColor(100, 100, 100);
        pdf.text('Map preview not available', pageWidth / 2, yPos + 25, { align: 'center' });
        yPos += 60;
    }

    // === Summary Statistics ===
    pdf.setFontSize(12);
    pdf.setTextColor(51, 51, 51);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Summary Statistics', margin, yPos);
    yPos += 8;

    // Summary box
    pdf.setFillColor(248, 250, 252);
    pdf.setDrawColor(200, 200, 200);
    pdf.roundedRect(margin, yPos, contentWidth, 35, 3, 3, 'FD');

    const improvement = data.optimizedScore - data.baselineScore;
    const improvementPct = data.baselineScore > 0
        ? ((improvement / data.baselineScore) * 100).toFixed(1)
        : '0.0';

    pdf.setFontSize(10);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(100, 100, 100);

    const col1X = margin + 10;
    const col2X = margin + contentWidth / 2;

    yPos += 10;
    pdf.text('Baseline Score:', col1X, yPos);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(51, 51, 51);
    pdf.text(data.baselineScore.toFixed(1), col1X + 40, yPos);

    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(100, 100, 100);
    pdf.text('Optimized Score:', col2X, yPos);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(16, 185, 129); // green
    pdf.text(data.optimizedScore.toFixed(1), col2X + 45, yPos);

    yPos += 10;
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(100, 100, 100);
    pdf.text('Improvement:', col1X, yPos);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(16, 185, 129);
    pdf.text(`+${improvement.toFixed(1)} (+${improvementPct}%)`, col1X + 40, yPos);

    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(100, 100, 100);
    pdf.text('Total Amenities:', col2X, yPos);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(51, 51, 51);
    pdf.text(data.amenities.length.toString(), col2X + 45, yPos);

    yPos += 20;

    // === Amenity Breakdown by Type ===
    const amenitiesByType = data.amenities.reduce((acc, a) => {
        const type = a.type.toLowerCase();
        acc[type] = (acc[type] || 0) + 1;
        return acc;
    }, {} as Record<string, number>);

    pdf.setFontSize(12);
    pdf.setTextColor(51, 51, 51);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Amenity Distribution', margin, yPos);
    yPos += 8;

    pdf.setFontSize(9);
    pdf.setFont('helvetica', 'normal');

    const typeEntries = Object.entries(amenitiesByType).sort((a, b) => b[1] - a[1]);
    const colWidth = contentWidth / 4;

    typeEntries.forEach((entry, i) => {
        const [type, count] = entry;
        const col = i % 4;
        const row = Math.floor(i / 4);
        const x = margin + col * colWidth;
        const y = yPos + row * 6;

        const color = AMENITY_COLORS[type] || '#666666';
        pdf.setFillColor(
            parseInt(color.slice(1, 3), 16),
            parseInt(color.slice(3, 5), 16),
            parseInt(color.slice(5, 7), 16)
        );
        pdf.circle(x + 2, y - 1.5, 1.5, 'F');

        pdf.setTextColor(51, 51, 51);
        const displayName = type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        pdf.text(`${displayName}: ${count}`, x + 6, y);
    });

    yPos += Math.ceil(typeEntries.length / 4) * 6 + 10;

    // === PAGE 2+: Detailed Amenity Table ===
    pdf.addPage();
    yPos = margin;

    // Table header
    pdf.setFillColor(27, 35, 40);
    pdf.rect(0, 0, pageWidth, 25, 'F');

    pdf.setTextColor(143, 214, 255);
    pdf.setFontSize(16);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Proposed Amenity Placements', margin, 16);

    yPos = 35;

    // Table column headers
    const colWidths = {
        type: 40,
        nodeId: 35,
        lat: 45,
        lon: 45,
    };

    pdf.setFillColor(240, 240, 240);
    pdf.rect(margin, yPos, contentWidth, 8, 'F');

    pdf.setFontSize(9);
    pdf.setTextColor(80, 80, 80);
    pdf.setFont('helvetica', 'bold');

    let xPos = margin + 3;
    pdf.text('Type', xPos, yPos + 5.5);
    xPos += colWidths.type;
    pdf.text('Node ID', xPos, yPos + 5.5);
    xPos += colWidths.nodeId;
    pdf.text('Latitude', xPos, yPos + 5.5);
    xPos += colWidths.lat;
    pdf.text('Longitude', xPos, yPos + 5.5);

    yPos += 10;
    pdf.setFont('helvetica', 'normal');

    // Sort amenities by type for better grouping
    const sortedAmenities = [...data.amenities].sort((a, b) =>
        a.type.localeCompare(b.type) || a.nodeId.localeCompare(b.nodeId)
    );

    sortedAmenities.forEach((amenity, index) => {
        // Check if we need a new page
        if (yPos > pageHeight - 20) {
            pdf.addPage();
            yPos = margin;

            // Repeat table header
            pdf.setFillColor(240, 240, 240);
            pdf.rect(margin, yPos, contentWidth, 8, 'F');

            pdf.setFontSize(9);
            pdf.setTextColor(80, 80, 80);
            pdf.setFont('helvetica', 'bold');

            let hxPos = margin + 3;
            pdf.text('Type', hxPos, yPos + 5.5);
            hxPos += colWidths.type;
            pdf.text('Node ID', hxPos, yPos + 5.5);
            hxPos += colWidths.nodeId;
            pdf.text('Latitude', hxPos, yPos + 5.5);
            hxPos += colWidths.lat;
            pdf.text('Longitude', hxPos, yPos + 5.5);

            yPos += 10;
            pdf.setFont('helvetica', 'normal');
        }

        // Alternating row colors
        if (index % 2 === 1) {
            pdf.setFillColor(248, 250, 252);
            pdf.rect(margin, yPos - 3, contentWidth, 7, 'F');
        }

        pdf.setFontSize(8);
        pdf.setTextColor(51, 51, 51);

        let rowX = margin + 3;

        // Type with color indicator
        const typeColor = AMENITY_COLORS[amenity.type.toLowerCase()] || '#666666';
        pdf.setFillColor(
            parseInt(typeColor.slice(1, 3), 16),
            parseInt(typeColor.slice(3, 5), 16),
            parseInt(typeColor.slice(5, 7), 16)
        );
        pdf.circle(rowX + 1, yPos + 0.5, 1.5, 'F');

        const displayType = amenity.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        pdf.text(displayType, rowX + 5, yPos + 2);
        rowX += colWidths.type;

        pdf.setFont('helvetica', 'normal');
        pdf.text(amenity.nodeId, rowX, yPos + 2);
        rowX += colWidths.nodeId;

        pdf.text(amenity.latitude.toFixed(6), rowX, yPos + 2);
        rowX += colWidths.lat;

        pdf.text(amenity.longitude.toFixed(6), rowX, yPos + 2);

        yPos += 7;
    });

    // Footer on last page
    yPos = pageHeight - 10;
    pdf.setFontSize(8);
    pdf.setTextColor(150, 150, 150);
    pdf.text(
        `Generated by PathLens • ${dateStr}`,
        pageWidth / 2,
        yPos,
        { align: 'center' }
    );

    // Save the PDF
    const filename = `pathlens-report-${data.location?.toLowerCase().replace(/[^a-z0-9]/g, '-') || 'optimization'}-${data.generatedAt.toISOString().slice(0, 10)
        }.pdf`;

    pdf.save(filename);
}

/**
 * Transform suggestions from the API into AmenityPlacement format
 */
export function transformSuggestionsToAmenities(
    suggestions: Array<{
        properties?: {
            id?: string;
            amenity_type?: string;
            amenity?: string;
        };
        geometry?: {
            coordinates?: [number, number];
        };
    }>
): AmenityPlacement[] {
    return suggestions
        .filter(s => s.geometry?.coordinates && s.properties)
        .map(s => ({
            type: s.properties?.amenity_type || s.properties?.amenity || 'unknown',
            nodeId: s.properties?.id || 'N/A',
            longitude: s.geometry!.coordinates![0],
            latitude: s.geometry!.coordinates![1],
        }));
}
