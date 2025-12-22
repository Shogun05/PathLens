import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Map } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || '';

export const MapEmbedding = ({ timestamp }) => {
  const mapUrl = `${BACKEND_URL}/outputs/optimized_map.html?run=${timestamp}`;

  return (
    <Card className="border-border">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-success/10 text-success">
            <Map className="h-5 w-5" />
          </div>
          <div>
            <CardTitle className="text-lg">Optimized Layout Map</CardTitle>
            <CardDescription className="text-xs">
              Interactive map showing the optimized urban layout. Pan, zoom, and click to explore.
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="map-container">
          <iframe
            src={mapUrl}
            title="Optimized Map"
            allow="geolocation"
            loading="lazy"
          />
        </div>
        <p className="mt-4 text-xs text-muted-foreground text-center">
          Use mouse to pan and zoom. Scroll wheel zooms when cursor is over the map.
        </p>
      </CardContent>
    </Card>
  );
};

export default MapEmbedding;
