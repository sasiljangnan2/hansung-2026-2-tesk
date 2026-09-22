//클라이언트는 메시지를 받기만 하도록 구현
//앞의 코드를 다음과 같이 수정, 이후 Ctrl+Shift+O 눌러서 import 파일 정리

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.Socket;
import java.util.Scanner;

public class TestClient2 {
	public static void main(String[] args) {
		
		try {
			Socket socket = new Socket("localhost", 9999); // 클라이언트 소켓 생성. 서버에 바로 접속
			System.out.println("클라이언트입니다. 서버에 접속하였습니다.");
			
			BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream())); 
			
			Scanner scanner = new Scanner(System.in);
			
			String inputMessage;
			
			while (true) {
		
				inputMessage = in.readLine(); // 서버로부터 한 행의 텍스트 받음
				System.out.println("받은 메시지: " + inputMessage); // 서버가 보낸 메시지 화면에 출력
				if (inputMessage.equals("끝")) { // 서버가 "끝"을 보내면 연결 종료
					System.out.println("연결을 종료합니다.");
					break;
				}	
			}
			
			scanner.close();
			socket.close(); 
			
		} catch (IOException e) {
			System.out.println("오류가 발생했습니다.");
		}
	}

}


