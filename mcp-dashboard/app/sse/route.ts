export const runtime = 'edge';

const encoder = new TextEncoder();

export async function GET(request: Request) {
  const { signal } = request;

  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode('data: connected\n\n'));
      const id = setInterval(() => {
        controller.enqueue(encoder.encode('data: ping\n\n'));
      }, 15000);

      signal.addEventListener('abort', () => {
        clearInterval(id);
        try { controller.close(); } catch (e) {}
      });
    }
  });

  return new Response(stream, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive'
    }
  });
}
