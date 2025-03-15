
// Proctoring utilities to detect suspicious behavior

// Key events tracking
let keyEvents: { key: string; timestamp: number }[] = [];
const keyPressThreshold = 120; // Characters per minute (suspiciously fast)

// Mouse events tracking
let mouseEvents: { x: number; y: number; timestamp: number }[] = [];
const mouseMovementThreshold = 50; // Unusual mouse movements

// Focus and visibility tracking
let focusEvents: { type: 'focus' | 'blur' | 'visibilitychange'; timestamp: number }[] = [];

// Initialize proctoring
export const initProctoring = (options: { 
  monitorKeystrokes?: boolean;
  monitorMouseMovements?: boolean;
  preventTabSwitching?: boolean;
  preventCopyPaste?: boolean;
  onSuspiciousActivity?: (eventType: string, data: any) => void;
}) => {
  const {
    monitorKeystrokes = true,
    monitorMouseMovements = true,
    preventTabSwitching = true,
    preventCopyPaste = true,
    onSuspiciousActivity = () => {}
  } = options;

  // Reset tracking arrays
  keyEvents = [];
  mouseEvents = [];
  focusEvents = [];
  
  // Monitor keystrokes
  if (monitorKeystrokes) {
    document.addEventListener('keydown', (e) => {
      keyEvents.push({ key: e.key, timestamp: Date.now() });
      
      // Check for suspicious typing patterns (very fast typing bursts)
      const recentKeyEvents = keyEvents.filter(
        event => Date.now() - event.timestamp < 10000 // Last 10 seconds
      );
      
      if (recentKeyEvents.length > 20) {
        const typingSpeed = calculateTypingSpeed(recentKeyEvents);
        if (typingSpeed > keyPressThreshold) {
          onSuspiciousActivity('irregular_keystrokes', { 
            typingSpeed,
            threshold: keyPressThreshold
          });
        }
      }
    });
  }
  
  // Monitor mouse movements
  if (monitorMouseMovements) {
    document.addEventListener('mousemove', (e) => {
      mouseEvents.push({ 
        x: e.clientX, 
        y: e.clientY, 
        timestamp: Date.now() 
      });
      
      // Check for suspicious mouse patterns (erratic movements)
      const recentMouseEvents = mouseEvents.filter(
        event => Date.now() - event.timestamp < 5000 // Last 5 seconds
      );
      
      if (recentMouseEvents.length > 10) {
        const mouseMovementScore = calculateMouseMovementScore(recentMouseEvents);
        if (mouseMovementScore > mouseMovementThreshold) {
          onSuspiciousActivity('irregular_mouse', { 
            score: mouseMovementScore,
            threshold: mouseMovementThreshold
          });
        }
      }
      
      // Trim the array to prevent memory issues
      if (mouseEvents.length > 100) {
        mouseEvents = mouseEvents.slice(-100);
      }
    });
  }
  
  // Monitor tab switching and window focus
  if (preventTabSwitching) {
    window.addEventListener('blur', () => {
      focusEvents.push({ type: 'blur', timestamp: Date.now() });
      onSuspiciousActivity('tab_switch', { 
        action: 'blur',
        timestamp: Date.now()
      });
    });
    
    window.addEventListener('focus', () => {
      focusEvents.push({ type: 'focus', timestamp: Date.now() });
    });
    
    document.addEventListener('visibilitychange', () => {
      focusEvents.push({ type: 'visibilitychange', timestamp: Date.now() });
      if (document.hidden) {
        onSuspiciousActivity('tab_switch', { 
          action: 'visibility_change',
          timestamp: Date.now()
        });
      }
    });
  }
  
  // Prevent copy and paste
  if (preventCopyPaste) {
    document.addEventListener('copy', (e) => {
      e.preventDefault();
      onSuspiciousActivity('copy_paste', { 
        action: 'copy',
        timestamp: Date.now()
      });
    });
    
    document.addEventListener('paste', (e) => {
      e.preventDefault();
      onSuspiciousActivity('copy_paste', { 
        action: 'paste',
        timestamp: Date.now()
      });
    });
    
    document.addEventListener('cut', (e) => {
      e.preventDefault();
      onSuspiciousActivity('copy_paste', { 
        action: 'cut',
        timestamp: Date.now()
      });
    });
  }
  
  return {
    stopProctoring: () => {
      // Clean up event listeners
      document.removeEventListener('keydown', () => {});
      document.removeEventListener('mousemove', () => {});
      window.removeEventListener('blur', () => {});
      window.removeEventListener('focus', () => {});
      document.removeEventListener('visibilitychange', () => {});
      document.removeEventListener('copy', () => {});
      document.removeEventListener('paste', () => {});
      document.removeEventListener('cut', () => {});
    }
  };
};

// Calculate typing speed in characters per minute
const calculateTypingSpeed = (events: { key: string; timestamp: number }[]) => {
  if (events.length < 2) return 0;
  
  const firstTimestamp = events[0].timestamp;
  const lastTimestamp = events[events.length - 1].timestamp;
  const durationMinutes = (lastTimestamp - firstTimestamp) / 60000;
  
  return events.length / Math.max(durationMinutes, 0.1); // Avoid division by very small numbers
};

// Calculate a score for mouse movement irregularity
const calculateMouseMovementScore = (events: { x: number; y: number; timestamp: number }[]) => {
  if (events.length < 3) return 0;
  
  let score = 0;
  
  for (let i = 2; i < events.length; i++) {
    const pt1 = events[i - 2];
    const pt2 = events[i - 1];
    const pt3 = events[i];
    
    // Calculate change in direction
    const vector1x = pt2.x - pt1.x;
    const vector1y = pt2.y - pt1.y;
    const vector2x = pt3.x - pt2.x;
    const vector2y = pt3.y - pt2.y;
    
    // Simple movement irregularity score
    const directionChange = Math.abs(
      Math.atan2(vector1y, vector1x) - Math.atan2(vector2y, vector2x)
    );
    
    // Timing between movements
    const timeDiff1 = pt2.timestamp - pt1.timestamp;
    const timeDiff2 = pt3.timestamp - pt2.timestamp;
    const timeIrregularity = Math.abs(timeDiff1 - timeDiff2);
    
    score += directionChange * timeIrregularity / 100;
  }
  
  return score / events.length;
};

// Take a screenshot of the current page (in a real implementation, this would use the
// browser's screenshot API or a screen capture library)
export const captureScreenshot = async (): Promise<string> => {
  // Mock implementation - in a real app, this would capture actual screenshots
  console.log('Screenshot captured');
  return 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z/C/HgAGgwJ/lK3Q6wAAAABJRU5ErkJggg==';
};
