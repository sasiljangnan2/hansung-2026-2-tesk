import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class TestKeyboard2 {
    public static void main(String[] args) {

        try {
            InputStream is = System.in;
            InputStreamReader rd = new InputStreamReader(is);

            System.out.print("텍스트 입력: ");
            
            char[] cbuf = new char[1024];
            int charRead = rd.read(cbuf);

            String inputMessage = new String(cbuf, 0, charRead);
            System.out.print(inputMessage);

        } catch (IOException e) {
            System.out.println("오류가 발생했습니다.");
        }
    }
}


