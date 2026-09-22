import java.util.Scanner;

public class TestKeyboard4 {
    public static void main(String[] args) {


        	Scanner scanner = new Scanner(System.in); 

            System.out.print("텍스트 입력: ");
            
        	String inputMessage = scanner.nextLine(); // 서버로부터 한 행의 텍스트 받음
			System.out.println(inputMessage); // 서버가 보낸 메시지 화면에 출력

			scanner.close();
			
    }
}


