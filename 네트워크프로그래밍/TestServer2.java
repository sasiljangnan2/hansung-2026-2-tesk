//서버는 메시지를 보내기만 하도록 구현, 서버에서 "끝"을 보내면 연결 종료
//앞의 코드를 다음과 같이 수정, 이후 Ctrl+Shift+O 눌러서 import 파일 정리

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Scanner;

public class TestServer2 {
	public static void main(String[] args) {
		try {

			ServerSocket listener = new ServerSocket(9999); // 서버 소켓 생성
			
			System.out.println("서버입니다. 클라이언트를 기다립니다.");
			Socket socket = listener.accept(); // 클라이언트로부터 연결 요청 대기
			System.out.println("클라이언트와 연결되었습니다.");
			
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
			
			Scanner scanner = new Scanner(System.in);
			
			String outputMessage;
			
			while (true) {
				
				System.out.print("보낼 메시지 입력 >> ");
				outputMessage = scanner.nextLine(); // 키보드에서 한 행의 문자열 읽음
				out.write(outputMessage+"\n"); // 클라이언트로 보냄
				out.flush();
				if (outputMessage.equals("끝")) { // "끝"이 입력되면 서버와 연결 종료
					System.out.println("연결을 종료합니다.");
					break;
				}
			}
			
			scanner.close();
			socket.close(); 
			listener.close(); 
			
		} catch (IOException e) {
			System.out.println("오류가 발생했습니다.");
		}
	}
}


