package prediction;

import java.net.URISyntaxException;

import org.json.JSONObject;

import io.socket.client.IO;
import io.socket.client.Socket;
import io.socket.emitter.Emitter;

public class EmitData {
	private NetFactory nnf;
	private Socket socket1 ;
	public EmitData() {
		 nnf = new NetFactory();
		 nnf.loadModelFromFile("CLOSE.zip");
		 
		
		try {

			 socket1 = IO.socket("http://localhost:8080");
			JSONObject obj = new JSONObject();

				
			socket1.connect();
			socket1.on("message",new Emitter.Listener() {

					public void call(Object... O) {
						// TODO Auto-generated method stub
						String ans  = (String) O[0];
						System.out.println("j'ai recu un message !"+ans);
						socket1.emit("message", nnf.predictObject(nnf.getMyIter().getTrain(), PriceCategory.CLOSE,"AAPL"));
						
					}});

		} catch (URISyntaxException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}

}
