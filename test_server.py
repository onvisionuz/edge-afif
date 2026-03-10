from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        print("\n--- Received Event Batch ---")
        try:
            data = json.loads(post_data.decode('utf-8'))
            print(f"Batch size: {len(data)} events")
            if data:
                print("First event preview:", json.dumps(data[0], indent=2))
        except Exception as e:
            print("Failed to parse JSON:", e)
        print("----------------------------\n")
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

if __name__ == '__main__':
    port = 8085
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Starting dummy event server on port {port}...")
    httpd.serve_forever()
