//(클라이언트와 연결이 되면) "연결 완료" 메시지를 한 번만 자동으로 보내고 종료(가장 심플하게 구현)
//앞의 코드를 다음과 같이 수정, 이후 Ctrl+Shift+O 눌러서 import 파일 정리

import java.io.DataOutputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.BindException;

public class TestServer5 {
	public static void main(String[] args) {
		try (ServerSocket listener = new ServerSocket(9999)) { // 서버 소켓 생성
			System.out.println("서버 실행 중: 9999번 포트에서 클라이언트 접속을 기다립니다.");
			try (Socket socket = listener.accept()) { // 클라이언트로부터 연결 요청 대기
			
			DataOutputStream os = new DataOutputStream(socket.getOutputStream());
			os.writeUTF("서버 연결 완료");
		

			}
			
		} catch (BindException e) {
			System.err.println("9999번 포트를 열 수 없습니다. 이미 실행 중인 서버가 있다면 먼저 종료하세요.");
			System.err.println("상세 오류: " + e);
		} catch (IOException e) {
			System.err.println("서버 실행 또는 통신 중 오류: " + e);
		}
	}
}

