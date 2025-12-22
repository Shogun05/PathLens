# PathLens - Urban Layout Optimization UI

A professional, single-page frontend interface for configuring and running urban layout optimization processes with real-time logging and interactive map visualization.

## Features

### 🎛️ Parameter Configuration
- **Composite Score Weights**: Configure α (Structure), β (Accessibility), γ (Equity), and δ (Travel Time) weights
- **Amenity Accessibility**: Set importance levels for hospitals, schools, groceries, transit, parks, and pharmacies
- **Distance Thresholds**: Define maximum walking distance and amenity search radius
- **Performance Tuning**: Adjust H3 resolution, centrality sample size, and walking speed

### 🚀 Optimization Process
- One-click optimization trigger with "Generate Optimized Layout" button
- Real-time backend log streaming via polling (500ms intervals)
- Progress tracking with detailed timestamped logs
- Automatic UI state management (disabled controls during processing)

### 📊 Real-Time Logging
- Fixed-height log panel (280px) with auto-scroll functionality
- Color-coded log levels:
  - **INFO**: Cyan (teal)
  - **WARNING**: Amber
  - **ERROR/CRITICAL**: Red
- Monospace font for technical readability
- Manual scroll control with auto-scroll resume

### 🗺️ Interactive Map Display
- Embedded Leaflet-based interactive map
- Full pan, zoom, and click functionality
- Shows optimized amenity locations with color-coded markers
- Walkability zones with heat overlay
- Legend for amenity types
- 700px fixed height with proper scroll handling

## Design System

### Color Palette
- **Primary**: Deep blue (HSL: 210 90% 48%) - Professional, trustworthy
- **Accent**: Teal (HSL: 180 65% 50%) - Data, technology
- **Success**: Green (HSL: 145 65% 45%) - Completion
- **Warning**: Amber (HSL: 38 92% 50%) - Alerts
- **Background**: Light slate (HSL: 220 18% 97%)

### Typography
- **Headings & Body**: Inter (Google Fonts)
- **Monospace/Logs**: IBM Plex Mono
- Clean, technical aesthetic for research/demo use

### Components
All UI components built with **shadcn/ui** for:
- Accessibility (WCAG AA compliant)
- Consistency across the application
- Professional appearance
- Smooth transitions and interactions

## Technical Architecture

### Frontend Stack
- **React 19**: Modern component-based architecture
- **Tailwind CSS**: Utility-first styling with design tokens
- **Shadcn/UI**: High-quality accessible components
- **Axios**: HTTP client for API communication
- **Sonner**: Toast notifications

### Backend Integration
- **POST /api/run-optimization**: Trigger optimization with parameters
- **GET /api/logs/latest**: Poll for real-time logs (500ms interval)
- **GET /outputs/optimized_map.html**: Serve generated interactive map

### State Management
- Local React state for parameters and UI state
- Polling-based log updates (more reliable than SSE for this use case)
- Automatic cleanup on component unmount

## User Flow

1. **Configure Parameters**: Adjust sliders to set optimization criteria
2. **Start Optimization**: Click "Generate Optimized Layout" button
3. **Monitor Progress**: Watch real-time logs as backend processes data
4. **View Results**: Interact with the generated optimization map
5. **Iterate**: Adjust parameters and run again for comparison

## Key Requirements Met

✅ **Single-page UI** with no reloads  
✅ **Slider-based parameter configuration** with live value display  
✅ **Real-time backend log display** via polling  
✅ **Fixed-height log panel** (280px) with auto-scroll  
✅ **Interactive map embedding** in iframe  
✅ **Full map interactivity** (pan, zoom, click, scroll)  
✅ **Proper scroll conflict handling** between page and map  
✅ **Professional design** suitable for research/demo  
✅ **Stable for long-running jobs** with proper state management  

## Mock Functionality

This is a **frontend prototype** with simulated backend processes:
- Optimization runs a ~30-second simulation with realistic log messages
- The backend generates a demo Leaflet map with sample data (Bangalore example)
- All interactions and UI flows work as they would in a production system
- Parameters are accepted and displayed in logs but don't affect the demo output

## Running the Application

```bash
# Backend is running on port 8001 (internal)
# Frontend is running on port 3000
# Access the application at: http://localhost:3000
```

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari
- Supports modern ES6+ features

## Design Philosophy

**Professional, Data-Driven, Trustworthy**

The design emphasizes:
- Clear information hierarchy
- Sufficient whitespace for readability
- Technical professionalism without being intimidating
- Smooth interactions and transitions
- Accessibility as a core principle
- Consistent use of design tokens (no hardcoded colors)

---

**Built with attention to detail for urban planning research and demonstration.**
