'use client';

import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { ArrowRight, Map, BarChart3, Sparkles, Target, Zap, Globe } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function LandingPage() {
  const router = useRouter();

  const features = [
    {
      icon: Map,
      title: 'Spatial Intelligence',
      description: 'Analyze urban networks using advanced graph algorithms and spatial data',
    },
    {
      icon: Target,
      title: 'Optimal Placement',
      description: 'AI-powered recommendations for amenity locations to maximize accessibility',
    },
    {
      icon: BarChart3,
      title: 'Impact Metrics',
      description: 'Measure and visualize improvements in travel time and equity scores',
    },
    {
      icon: Zap,
      title: 'Real-time Optimization',
      description: 'Genetic algorithms combined with MILP for fast, accurate solutions',
    },
  ];

  return (
    <div className="min-h-screen bg-[#0f1c23] text-white overflow-hidden">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[#0f1c23]/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="size-10 flex items-center justify-center rounded-lg bg-gradient-to-br from-[#8fd6ff] to-[#5fb3e6] text-[#0f1c23]">
              <Globe className="h-5 w-5" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-[#8fd6ff] to-white bg-clip-text text-transparent">
              PathLens
            </span>
          </div>
          <div className="flex items-center gap-4">
            <Button variant="ghost" className="text-gray-400 hover:text-white hidden sm:inline-flex">
              Documentation
            </Button>
            <Button variant="ghost" className="text-gray-400 hover:text-white hidden sm:inline-flex">
              About
            </Button>
            <Button
              onClick={() => router.push('/setup')}
              className="bg-[#8fd6ff] hover:bg-[#b0e2ff] text-[#0f1c23] font-semibold"
            >
              Get Started
            </Button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-6">
        {/* Background Effects */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-20 left-1/4 w-96 h-96 bg-[#8fd6ff]/10 rounded-full blur-3xl" />
          <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-[#5fb3e6]/10 rounded-full blur-3xl" />
          <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-5" />
        </div>

        <div className="relative max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center max-w-4xl mx-auto"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#8fd6ff]/10 border border-[#8fd6ff]/20 text-[#8fd6ff] text-sm font-medium mb-8">
              <Sparkles className="h-4 w-4" />
              AI-Powered Urban Planning
            </div>
            
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold leading-tight mb-6">
              <span className="text-white">Optimize Urban</span>
              <br />
              <span className="bg-gradient-to-r from-[#8fd6ff] via-[#5fb3e6] to-[#8fd6ff] bg-clip-text text-transparent">
                Accessibility
              </span>
            </h1>
            
            <p className="text-lg sm:text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed">
              Transform city planning with intelligent amenity placement. 
              PathLens uses advanced optimization algorithms to improve 
              accessibility and reduce travel times across urban networks.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Button
                onClick={() => router.push('/setup')}
                size="lg"
                className="h-14 px-8 bg-[#8fd6ff] hover:bg-[#b0e2ff] text-[#0f1c23] font-bold text-lg shadow-[0_0_30px_rgba(143,214,255,0.3)] hover:shadow-[0_0_40px_rgba(143,214,255,0.4)] transition-all"
              >
                Get Started
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button
                variant="outline"
                size="lg"
                className="h-14 px-8 border-white/20 text-white hover:bg-white/10 font-semibold text-lg"
              >
                View Demo
              </Button>
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto"
          >
            {[
              { value: '3+', label: 'Cities Analyzed' },
              { value: '77K+', label: 'Network Nodes' },
              { value: '95%', label: 'Equity Score' },
              { value: '30%', label: 'Travel Time Reduced' },
            ].map((stat, i) => (
              <div key={i} className="text-center">
                <div className="text-3xl sm:text-4xl font-bold text-[#8fd6ff] mb-1">{stat.value}</div>
                <div className="text-sm text-gray-500">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-6 bg-[#0a1419]">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Powerful <span className="text-[#8fd6ff]">Optimization</span> Engine
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Leverage cutting-edge algorithms to make data-driven decisions 
              about urban infrastructure placement.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, i) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                viewport={{ once: true }}
                className="group p-6 rounded-2xl bg-[#1b2328] border border-white/5 hover:border-[#8fd6ff]/30 transition-all"
              >
                <div className="size-12 flex items-center justify-center rounded-xl bg-[#8fd6ff]/10 text-[#8fd6ff] mb-4 group-hover:bg-[#8fd6ff] group-hover:text-[#0f1c23] transition-all">
                  <feature.icon className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                <p className="text-sm text-gray-400 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="p-10 rounded-3xl bg-gradient-to-br from-[#1b2328] to-[#0f1c23] border border-white/10"
          >
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Ready to optimize your city?
            </h2>
            <p className="text-gray-400 mb-8 max-w-xl mx-auto">
              Start analyzing your urban network and discover optimal 
              locations for new amenities in minutes.
            </p>
            <Button
              onClick={() => router.push('/setup')}
              size="lg"
              className="h-14 px-10 bg-[#8fd6ff] hover:bg-[#b0e2ff] text-[#0f1c23] font-bold text-lg shadow-[0_0_30px_rgba(143,214,255,0.3)]"
            >
              Get Started Now
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-6 border-t border-white/5">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <Globe className="h-4 w-4" />
            <span>©PathLens 2025</span>
          </div>
          <div className="flex items-center gap-6 text-sm text-gray-500">
            <a href="#" className="hover:text-white transition-colors">Documentation</a>
            <a href="#" className="hover:text-white transition-colors">GitHub</a>
            <a href="#" className="hover:text-white transition-colors">About</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
