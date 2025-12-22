import React from 'react';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { Play, Loader2 } from 'lucide-react';

export const ActionControl = ({ onStart, disabled, isRunning }) => {
  return (
    <Card className="border-border">
      <CardContent className="py-8">
        <div className="flex flex-col items-center gap-4">
          <div className="text-center space-y-2">
            <h3 className="text-lg font-semibold text-foreground">Ready to Optimize</h3>
            <p className="text-sm text-muted-foreground max-w-2xl">
              Click the button below to start the optimization process with your configured parameters.
              The process may take several minutes depending on the complexity of the area.
            </p>
          </div>
          
          <Button
            onClick={onStart}
            disabled={disabled}
            size="lg"
            className="h-12 px-8 text-base font-semibold smooth-transition"
          >
            {isRunning ? (
              <>
                <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                Optimizing...
              </>
            ) : (
              <>
                <Play className="mr-2 h-5 w-5" />
                Generate Optimized Layout
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default ActionControl;
