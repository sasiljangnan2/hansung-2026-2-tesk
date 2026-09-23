import java.io.*;
import java.net.*;
import java.util.*;

public class ServerEx {
	public static void main(String[] args) {
		BufferedReader in = null;
		BufferedWriter out = null;
		ServerSocket listener = null;
		Socket socket = null;
		Scanner scanner = new Scanner(System.in, "CP949"); // 키보드에서 읽을 scanner 객체 생성
		try {
			listener = new ServerSocket(9999); // 서버 소켓 생성
			System.out.println("연결을 기다리고 있습니다.....");
			socket = listener.accept(); // 클라이언트로부터 연결 요청 대기
			System.out.println("연결되었습니다.");
			in = new BufferedReader(new InputStreamReader(socket.getInputStream())); // 소켓 입력 스트림
			out = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream())); // 소켓 출력 스트림
			// 수신은 별도 스레드에서 실행하고, 메인 스레드는 송신을 담당
			final BufferedReader receiveIn = in;
			final Socket receiveSocket = socket;
			Runnable receiver = new Runnable() {
				@Override
				public void run() {
					try {
						while (true) {
							String inputMessage = receiveIn.readLine(); // 상대방으로부터 한 행 수신
							if (inputMessage == null || inputMessage.equalsIgnoreCase("bye") || inputMessage.equalsIgnoreCase("끝")) {
								System.out.println("접속을 종료합니다.");
								break;
							}
							System.out.println("클라이언트: " + inputMessage); // 받은 메시지를 화면에 출력
							System.out.print(">>");
						}
					} catch (IOException e) {
						if (!receiveSocket.isClosed()) System.out.println(e.getMessage());
					} finally {
						try {
							receiveSocket.close();
						} catch (IOException e) {
							System.out.println(e.getMessage());
						}
						// 메인 스레드가 키보드 입력 대기 중이어도 프로그램 종료
						System.exit(0);
					}
				}
			};
			Thread receiveThread = new Thread(receiver);
			receiveThread.start();
			while (true) {
				System.out.print(">>");
				String outputMessage = scanner.nextLine(); // 키보드에서 한 행의 문자열 읽음
				out.write(outputMessage + "\n"); // 키보드에서 읽은 문자열 전송
				out.flush();
				if (outputMessage.equalsIgnoreCase("bye") || outputMessage.equalsIgnoreCase("끝")) {
					System.out.println("연결을 종료합니다.");
					break;
				}
			}
		} catch (IOException e) {
			System.out.println(e.getMessage());
		} finally {
			try {
				scanner.close(); // scanner 닫기
				if(socket != null) socket.close(); // 통신용 소켓 닫기
				if(listener != null) listener.close(); // 서버 소켓 닫기
			} catch (IOException e) {
				System.out.println("클라이언트와 채팅 중 오류가 발생했습니다.");
			}
		}
	}
}
