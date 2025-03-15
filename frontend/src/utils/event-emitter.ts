import { io, Socket } from 'socket.io-client';

export class KeyboardEventStreamer {
    private socket: Socket | null = null;
    private eventQueue: Array<{ type: 'KU' | 'KD', key: string, timestamp: number }> = [];
    private isConnected = false;

    constructor(private serverUrl: string) {
        this.serverUrl = serverUrl;
        this.connect();
    }

    private connect() {
        this.socket = io(this.serverUrl, {
            transports: ['polling', 'websocket'],
            reconnection: true,
            reconnectionAttempts: 5,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            withCredentials: false
        });

        this.socket.on('connect', () => {
            console.log('Socket.IO connection established');
            this.isConnected = true;

            if (this.eventQueue.length > 0) {
                this.flushQueue();
            }
        });

        this.socket.on('disconnect', () => {
            this.isConnected = false;
            console.log('Socket.IO disconnected');
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
        });
    }

    public sendEvent(type: 'KU' | 'KD', key: string) {
        const timestamp = Date.now(); // Use milliseconds for simplicity
        const event = { type, key, timestamp };

        if (this.isConnected && this.socket?.connected) {
            this.socket.emit('keyevents', [event]);
        } else {
            this.eventQueue.push(event);
        }
    }

    private flushQueue() {
        if (this.isConnected && this.socket?.connected) {
            while (this.eventQueue.length > 0) {
                const batch = this.eventQueue.splice(0, 50);
                this.socket.emit('keyevents', batch);
            }
        }
    }

    public disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
    }
}