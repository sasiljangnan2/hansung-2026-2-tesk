import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ServerM {

	//서버 종료코드는 없이 간단히 작성, 필요하면 별도의 서버 스레드를 만들어 "exit" 이력시 종료하는 코드 작성 가능
	
	static ArrayList<ServerThread> list = new ArrayList<>();    

    static int clientCount = 0;

    public static void main(String[] args) throws IOException {
        ServerSocket ssocket = new ServerSocket(5000);
        System.out.println("서버가 시작되었습니다. 포트: 5000");

        Socket s;

        while (true) {
            s = ssocket.accept();
            System.out.println("클라이언트 " + clientCount + " 연결됨");

            ServerThread thread = new ServerThread(s, "client " + clientCount);
            list.add(thread);
            thread.start();
            clientCount++;
        }
    }
}

class ServerThread extends Thread {
    private String name;
    private DataInputStream is;
    private DataOutputStream os;
    private Socket s;

    public ServerThread(Socket s, String name) {
        this.name = name;
        this.s = s;

        try {
            is = new DataInputStream(s.getInputStream());
            os = new DataOutputStream(s.getOutputStream());
        } catch (Exception e) {
            System.out.println(name + " 스트림 생성 중 예외 발생: " + e.getMessage());
        }
    }

    @Override
    public void run() {
        String message;
        try {
            while (true) {
                message = is.readUTF();
                System.out.println(name + "로부터 수신: " + message);

                // 모든 클라이언트에게 브로드캐스트
                WriteAll(this.name + " : " + message);
            }
        } catch (IOException e) {
            System.out.println(name + " 연결 종료");
        } finally {
            // 리스트에서 제거
            ServerM.list.remove(this);
            System.out.println(name + " 리스트에서 제거됨. 남은 클라이언트: " + ServerM.list.size());
            
            // 리소스 정리
            closeResources();
        }
    }

  //모든 다중 클라이언트에게 순차적으로 채팅 메시지 전달
    public void WriteAll(String str) {  
        for (int i = 0; i < ServerM.list.size(); i++) {
            ServerThread user = ServerM.list.get(i);  // i번째 사용자
            try {
                user.os.writeUTF(str);
                user.os.flush(); // 버퍼 비우기
            } catch (IOException e) {
                System.out.println(user.name + "에게 메시지 전송 실패");
            }
        }
    }


    // 리소스 정리
    private void closeResources() {
        try {
            if (is != null) is.close();
            if (os != null) os.close();
            if (s != null && !s.isClosed()) s.close();
        } catch (IOException e) {
            System.out.println(name + " 리소스 정리 중 예외: " + e.getMessage());
        }
    }
}