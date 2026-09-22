import java.io.*;
import java.net.*;
import java.util.*;

public class ClientEx {
	public static void main(String[] args) {
		BufferedReader in = null;
		BufferedWriter out = null;
		Socket socket = null;
		Scanner scanner = new Scanner(System.in, "CP949"); // 키보드에서 읽을 scanner 객체 생성
		try {
			socket = new Socket("localhost", 9999); // 클라이언트 소켓 생성. 서버와 바로 연결
			in = new BufferedReader(new InputStreamReader(socket.getInputStream())); // 소켓 입력 스트림
			out = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream())); // 소켓 출력 스트림
			while (true) {
				System.out.print("텍스트 입력>>"); // 프롬프트
				String outputMessage = scanner.nextLine(); // 키보드에서 한 행 읽기
				if (outputMessage.equalsIgnoreCase("bye") || outputMessage.equalsIgnoreCase("끝")) { // 사용자가 "bye", 끝을 입력하면 연결 종료
					System.out.println("연결을 종료합니다."); // 종료 메시지 출력
					out.write(outputMessage+"\n"); // "bye" 문자열 전송
					out.flush();
					break; // 사용자가 "bye"를 입력한 경우 서버로 전송 후 연결 종료
				}

				out.write(outputMessage + "\n"); // 키보드에서 읽은 문자열 전송
				out.flush();
				String inputMessage = in.readLine(); // 서버로부터 한 행 수신
				System.out.println("서버: " + inputMessage); // 서버로부터 받은 메시지를 화면에 출력
				//서버에서 클라이언트로 메시지를 보내는 부분은 주석처리해 메시지만 보내도록 수정
			}
		} catch (IOException e) {
			System.out.println(e.getMessage());
		} finally {
			try {
				scanner.close();
				if(socket != null) socket.close(); // 클라이언트 소켓 닫기
			} catch (IOException e) {
				System.out.println("서버와 채팅 중 오류가 발생했습니다.");
			}
		}
	}
}
