import java.io.IOException;
import java.io.InputStream;
import java.net.Socket;

public class TestClient32 {
    public static void main(String[] args) {

        try {
            Socket socket = new Socket("localhost", 9999); // 클라이언트 소켓 생성. 서버에 바로 접속

            InputStream is = socket.getInputStream();

            byte[] byteData = new byte[1024];
            int bytesRead = is.read(byteData);

            String inputMessage = new String(byteData, 0, bytesRead);
            System.out.println("받은 메시지: " + inputMessage);


            socket.close();

        } catch (IOException e) {
            System.out.println("오류가 발생했습니다.");
        }
    }
}


