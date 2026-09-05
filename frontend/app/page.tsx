import ChatUI from "@/components/ChatUI";

export default function Home() {
  return (
    <main className="flex flex-col h-screen">
      <header className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-lg font-semibold text-white">Voice Knowledge Agent</h1>
        <p className="text-xs text-gray-400">Ask questions about your audio library</p>
      </header>
      <div className="flex-1 overflow-hidden">
        <ChatUI />
      </div>
    </main>
  );
}
