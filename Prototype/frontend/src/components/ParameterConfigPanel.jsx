import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Label } from './ui/label';
import { Slider } from './ui/slider';
import { Scale, Building2, Ruler, Settings } from 'lucide-react';
import { parameterCategories } from '../config/parameters';

const iconMap = {
  Scale: Scale,
  Building2: Building2,
  Ruler: Ruler,
  Settings: Settings,
};

export const ParameterConfigPanel = ({ parameters, onParameterChange, disabled }) => {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h2 className="text-2xl font-semibold text-foreground">Optimization Parameters</h2>
        <p className="text-sm text-muted-foreground">
          Configure the parameters for the urban layout optimization process. Adjust sliders to fine-tune the optimization criteria.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {parameterCategories.map((category) => {
          const IconComponent = iconMap[category.icon];
          const categoryParams = parameters[category.key];

          return (
            <Card key={category.key} className="border-border hover:shadow-md smooth-transition">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <IconComponent className="h-5 w-5" />
                  </div>
                  <div>
                    <CardTitle className="text-lg">{category.title}</CardTitle>
                    <CardDescription className="text-xs">{category.description}</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {Object.entries(categoryParams).map(([key, param]) => (
                  <div key={key} className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label htmlFor={`${category.key}-${key}`} className="text-sm font-medium text-foreground">
                        {param.label}
                      </Label>
                      <span className="rounded-md bg-secondary px-2.5 py-1 text-sm font-semibold text-secondary-foreground">
                        {param.value}
                      </span>
                    </div>
                    <Slider
                      id={`${category.key}-${key}`}
                      min={param.min}
                      max={param.max}
                      step={param.step}
                      value={[param.value]}
                      onValueChange={(value) => onParameterChange(category.key, key, value[0])}
                      disabled={disabled}
                      className="w-full"
                    />
                    <p className="text-xs text-muted-foreground">{param.description}</p>
                  </div>
                ))}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
};

export default ParameterConfigPanel;
