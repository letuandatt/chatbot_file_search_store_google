"use client";

import type { ThinkingStep } from "@/types/chat";

interface ThinkingStepsProps {
    steps: ThinkingStep[];
    expanded: boolean;
    onToggle: () => void;
}

/**
 * Expandable timeline that visualises the multi-agent thinking
 * pipeline ("supervisor → tool_search_law → reranker → …"). The
 * parent owns the expanded/collapsed state so it can be persisted
 * per-message in a single Set without each step component
 * maintaining its own state.
 */
export function ThinkingSteps({ steps, expanded, onToggle }: ThinkingStepsProps) {
    if (!steps.length) return null;
    return (
        <div className="flex items-start gap-3 justify-start ml-11 mb-2">
            <div className="w-full max-w-[85%]">
                <button
                    onClick={onToggle}
                    className="group flex items-center gap-2 text-xs text-violet-400 hover:text-violet-300 transition-all duration-200 mb-2 py-1 px-2 -ml-2 rounded-lg hover:bg-violet-500/10"
                >
                    <svg
                        className={`w-3.5 h-3.5 transition-transform duration-200 ${expanded ? "rotate-90" : ""}`}
                        fill="currentColor"
                        viewBox="0 0 20 20"
                    >
                        <path
                            fillRule="evenodd"
                            d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                            clipRule="evenodd"
                        />
                    </svg>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={1.5}
                            d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                        />
                    </svg>
                    <span className="font-medium">Quá trình suy luận</span>
                    <span className="text-violet-500/70 text-[10px] bg-violet-500/20 px-1.5 py-0.5 rounded-full">
                        {steps.length}
                    </span>
                </button>

                {/* Timeline Content */}
                <div
                    className={`overflow-hidden transition-all duration-300 ease-in-out ${expanded ? "max-h-[500px] opacity-100" : "max-h-0 opacity-0"}`}
                >
                    <div className="bg-gradient-to-br from-violet-500/5 to-violet-600/10 border border-violet-500/20 rounded-xl p-4">
                        <div className="relative">
                            {/* Vertical timeline line */}
                            <div className="absolute left-[11px] top-3 bottom-3 w-0.5 bg-gradient-to-b from-violet-500/50 via-violet-400/30 to-violet-500/10 rounded-full"></div>

                            <div className="space-y-3">
                                {steps.map((step, stepIdx) => {
                                    const isLast = stepIdx === steps.length - 1;
                                    return (
                                        <div
                                            key={stepIdx}
                                            className="flex items-start gap-3 relative"
                                        >
                                            <div
                                                className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 z-10 transition-all duration-200 ${
                                                    isLast
                                                        ? "bg-violet-500 shadow-lg shadow-violet-500/30"
                                                        : "bg-violet-500/30 border border-violet-400/50"
                                                }`}
                                            >
                                                {isLast ? (
                                                    <svg
                                                        className="w-3 h-3 text-white"
                                                        fill="currentColor"
                                                        viewBox="0 0 20 20"
                                                    >
                                                        <path
                                                            fillRule="evenodd"
                                                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                                                            clipRule="evenodd"
                                                        />
                                                    </svg>
                                                ) : (
                                                    <span className="text-[10px] text-violet-300 font-semibold">
                                                        {stepIdx + 1}
                                                    </span>
                                                )}
                                            </div>
                                            <div className="flex-1 min-w-0 pb-1">
                                                <div className="flex items-center gap-2 flex-wrap">
                                                    <span className="text-xs font-semibold text-violet-300 bg-violet-500/20 px-2 py-0.5 rounded-md">
                                                        {step.agent}
                                                    </span>
                                                </div>
                                                <p className="text-xs text-zinc-400 mt-1 leading-relaxed">
                                                    {step.action}
                                                </p>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
