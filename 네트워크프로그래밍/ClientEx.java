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
			// 수신은 별도 스레드에서 실행하고, 메인 스레드는 송신을 담당
			Runnable receiver = new ClientReceiver(in, socket);
			Thread receiveThread = new Thread(receiver);
			receiveThread.start();
			while (true) {
				System.out.print(">>"); 
				String outputMessage = scanner.nextLine(); // 키보드에서 한 행 읽기
				if (outputMessage.equalsIgnoreCase("bye") || outputMessage.equalsIgnoreCase("끝")) { // 사용자가 "bye", 끝을 입력하면 연결 종료
					System.out.println("연결을 종료합니다."); // 종료 메시지 출력
					out.write(outputMessage+"\n"); // "bye" 문자열 전송
					out.flush();
					break; // 사용자가 "bye"를 입력한 경우 서버로 전송 후 연결 종료
				}

				out.write(outputMessage + "\n"); // 키보드에서 읽은 문자열 전송
				out.flush();
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

// 독립 클래스로 작성한 수신 작업
class ClientReceiver implements Runnable {
	private final BufferedReader receiveIn;
	private final Socket receiveSocket;

	public ClientReceiver(BufferedReader in, Socket socket) {
		this.receiveIn = in;
		this.receiveSocket = socket;
	}

	@Override
	public void run() {
		try {
			while (true) {
				String inputMessage = receiveIn.readLine(); // 상대방으로부터 한 행 수신
				if (inputMessage == null || inputMessage.equalsIgnoreCase("bye") || inputMessage.equalsIgnoreCase("끝")) {
					System.out.println("접속을 종료합니다.");
					break;
				}
				System.out.println("서버: " + inputMessage); // 받은 메시지를 화면에 출력
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
}
