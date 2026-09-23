import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class TestKeyboard3 {
    public static void main(String[] args) {

        try {
        	BufferedReader in = new BufferedReader(new InputStreamReader(System.in)); 

            System.out.print("텍스트 입력: ");
            
        	String inputMessage = in.readLine(); // 서버로부터 한 행의 텍스트 받음
			System.out.println(inputMessage); // 서버가 보낸 메시지 화면에 출력

        } catch (IOException e) {
            System.out.println("오류가 발생했습니다.");
        }
    }
}


