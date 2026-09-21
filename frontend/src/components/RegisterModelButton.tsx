"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { registerModel } from "@/lib/api";

export default function RegisterModelButton({ experimentId }: { experimentId: string }) {
  const [registering, setRegistering] = useState(false);
  const [done, setDone] = useState(false);
  const router = useRouter();

  async function handleRegister() {
    setRegistering(true);
    try {
      await registerModel(experimentId);
      setDone(true);
      router.refresh();
    } catch {
      setRegistering(false);
    }
  }

  if (done) {
    return <span className="text-green-600 text-xs font-medium">Registered</span>;
  }

  return (
    <button
      onClick={handleRegister}
      disabled={registering}
      className="text-xs bg-gray-800 text-white px-3 py-1 rounded disabled:bg-gray-300"
    >
      {registering ? "Registering..." : "Register"}
    </button>
  );
}