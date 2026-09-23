//서버 구현, 연결 후 클라이언트에서 먼저 메시지 보냄, 이후 순차적으로 한 번씩 메시지를 주고 받음
//서버와 클라이언트 중 어디서든 "끝"을 입력하면 서버와 클라이언트 모두 동시에 종료

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Scanner;

//소스 코드를 입력하고 Ctrl+Shift+O를 눌러서 필요한 파일을 포함 
public class TestServer {
	public static void main(String[] args) {
		try {

			ServerSocket listener = new ServerSocket(9999); // 서버 소켓 생성
			
			System.out.println("서버입니다. 클라이언트를 기다립니다.");			
			Socket socket = listener.accept(); // 클라이언트로부터 연결 요청 대기
			System.out.println("클라이언트와 연결되었습니다.");
			
			BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream())); 
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
			
			Scanner scanner = new Scanner(System.in);
			
			String inputMessage;
			String outputMessage;
			
			while (true) {
				inputMessage = in.readLine(); // 클라이언트로부터 한 행의 텍스트 받음
                                // 추가 설명: readLine()은 줄바꿈 문자가 들어올 때까지 문자를 계속 읽은 다음, 줄바꿈 문자를 제외한 문자열을 반환한다.
				System.out.println("받은 메시지: " + inputMessage); // 클라이언트가 보낸 메시지 화면에 출력
				if (inputMessage.equals("끝")) { // 클라이언트가 "끝"을 보내면 연결 종료, 참고: equalsIgnoreCase()는 대소문자를 구분하지 않고 비교
					System.out.println("연결을 종료합니다.");
					break;
				}			
				
				System.out.print("보낼 메시지 입력 >> ");
				outputMessage = scanner.nextLine(); // 키보드에서 한 행의 문자열 받음
				out.write(outputMessage+"\n"); // 클라이언트로 보냄
				out.flush();
				if (outputMessage.equals("끝")) { // "끝"이 입력되면 연결 종료
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


