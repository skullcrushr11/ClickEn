// fullscreenUtils.ts
import { Dispatch, SetStateAction } from 'react';
import { useToast } from '@/hooks/use-toast';

export const goFullScreen = async (setIsFullscreen: Dispatch<SetStateAction<boolean>>) => {
    const elem = document.documentElement as HTMLElement & {
        webkitRequestFullscreen?: () => Promise<void>;
        mozRequestFullScreen?: () => Promise<void>;
        msRequestFullscreen?: () => Promise<void>;
    };

    try {
        if (elem.requestFullscreen) {
            await elem.requestFullscreen();
        } else if (elem.webkitRequestFullscreen) {
            await elem.webkitRequestFullscreen();
        } else if (elem.mozRequestFullScreen) {
            await elem.mozRequestFullScreen();
        } else if (elem.msRequestFullscreen) {
            await elem.msRequestFullscreen();
        }
        setIsFullscreen(true);
        localStorage.setItem('fullscreenPreferred', 'true');
    } catch (err) {
        console.error("Fullscreen error:", err);
    }
};

export const handleCopy = async (e: ClipboardEvent, setInternalClipboard: Dispatch<SetStateAction<string>>) => {
    try {
        const selection = window.getSelection();
        if (selection && selection.toString().trim() !== '') {
            // Prevent the default copy behavior
            e.preventDefault();

            // Store the selected text in our internal clipboard
            const selectedText = selection.toString();
            setInternalClipboard(selectedText);

            // Optional: Show success feedback
            return `Copied text succesfully`
        }
    } catch (error) {
        // Handle any errors that might occur during selection or state update
        console.error("Error handling copy operation:", error);

        // Optional: Show error feedback to user

        // Try to avoid completely breaking functionality
        try {
            // Allow the default browser behavior as fallback
            setTimeout(() => {
                const tempSelection = window.getSelection();
                if (tempSelection && tempSelection.toString().trim() !== '') {
                    // Don't call preventDefault() here to allow native copy
                    console.log("Falling back to browser's default copy behavior");
                }
            }, 100);
        } catch (fallbackError) {
            console.error("Even fallback copy failed:", fallbackError);
        }
        return `Error copying text: ${error.message}`;
    }
}

export const handleCut = async (e: ClipboardEvent, setInternalClipboard: Dispatch<SetStateAction<string>>) => {
    try {
        const selection = window.getSelection();
        if (selection && selection.toString().trim() !== '') {
            const activeElement = document.activeElement as HTMLElement;
            const isEditableArea =
                activeElement &&
                (activeElement.tagName === 'INPUT' ||
                    activeElement.tagName === 'TEXTAREA' ||
                    activeElement.getAttribute('contenteditable') === 'true' ||
                    activeElement.closest('[contenteditable="true"]'));

            if (isEditableArea) {
                e.preventDefault();

                const selectedText = selection.toString();
                setInternalClipboard(selectedText);

                if ('selectionStart' in activeElement && 'selectionEnd' in activeElement) {
                    const start = (activeElement as HTMLInputElement).selectionStart;
                    const end = (activeElement as HTMLInputElement).selectionEnd;
                    const value = (activeElement as HTMLInputElement).value;

                    if (start !== null && end !== null) {
                        (activeElement as HTMLInputElement).value =
                            value.substring(0, start) + value.substring(end);
                        (activeElement as HTMLInputElement).selectionStart =
                            (activeElement as HTMLInputElement).selectionEnd = start;
                    }
                }
                else if (selection && selection.rangeCount > 0) {
                    selection.deleteFromDocument();
                }
            } else {
                e.preventDefault();
            }
        }
    } catch (error) {
        return `Error cutting text: ${error.message}`;
    }
};

export const handlePaste = (e: ClipboardEvent, internalClipboard: string) => {
    try {
        e.preventDefault();

        const activeElement = document.activeElement as HTMLElement;

        if (internalClipboard && activeElement) {
            try {
                if (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA') {
                    if ('selectionStart' in activeElement && 'selectionEnd' in activeElement) {
                        const start = (activeElement as HTMLInputElement).selectionStart;
                        const end = (activeElement as HTMLInputElement).selectionEnd;
                        const value = (activeElement as HTMLInputElement).value;

                        if (start !== null && end !== null) {
                            (activeElement as HTMLInputElement).value =
                                value.substring(0, start) + internalClipboard + value.substring(end);
                            const newCursorPos = start + internalClipboard.length;
                            (activeElement as HTMLInputElement).selectionStart =
                                (activeElement as HTMLInputElement).selectionEnd = newCursorPos;
                        }
                    }
                }
                else if (activeElement.getAttribute('contenteditable') === 'true' ||
                    activeElement.closest('[contenteditable="true"]')) {
                    const selection = window.getSelection();
                    if (selection && selection.rangeCount > 0) {
                        const range = selection.getRangeAt(0);
                        range.deleteContents();
                        range.insertNode(document.createTextNode(internalClipboard));
                        range.setStartAfter(range.endContainer);
                        range.collapse(true);
                        selection.removeAllRanges();
                        selection.addRange(range);
                    } else {
                        throw new Error("No valid selection range found");
                    }

                }
            } catch (pasteError) {
                console.error("Error pasting clipboard contents:", pasteError);
            }
        }
    } catch (error) {
        console.error("Critical error in paste handler:", error);
        try {
            return error.message;
        } catch (toastError) {
            return `Complete paste handling failure: ${toastError.message}`;
        }
    }
};