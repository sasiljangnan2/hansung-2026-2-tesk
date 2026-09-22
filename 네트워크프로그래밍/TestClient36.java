import java.io.DataInputStream;
import java.io.IOException;
import java.net.Socket;

public class TestClient36 {
    public static void main(String[] args) {

        try {
        	Socket socket = new Socket("localhost", 9999); // 클라이언트 소켓 생성. 서버에 바로 접속
        	
        	DataInputStream is = new DataInputStream(socket.getInputStream()); 
        	
        	String inputMessage = is.readUTF(); 
			System.out.println("받은 메시지: " + inputMessage); // 서버가 보낸 메시지 화면에 출력
			
            socket.close();
            
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
    }
}


