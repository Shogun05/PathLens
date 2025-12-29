'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function HomePage() {
  const router = useRouter();

  useEffect(() => {
    router.push('/setup');
  }, [router]);

  return (
    <div className="flex h-screen items-center justify-center bg-[#0f1c23]">
      <div className="flex items-center gap-4">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-[#8fd6ff] border-t-transparent"></div>
        <span className="text-[#8fd6ff] font-medium">Loading PathLens...</span>
      </div>
    </div>
  );
}
