package agents.anac.y2015.group2;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

class G2Logger {
	FileWriter fileWriter = null;
	PrintWriter printWriter;
	
	ArrayList<String> Buffer = new ArrayList<String>();
	
	G2Logger () {
		
	}
	
	void init (int partyNumber) {
		try {
			fileWriter = new FileWriter("./logs/party"+partyNumber+"log.txt");
			printWriter = new PrintWriter(fileWriter);
		} catch (IOException e) {
			System.out.println("Could not create log for party " + partyNumber);
		}
		printWriter.println("Log for party:" + partyNumber);
		printWriter.println("==========================================");
		for(String s: Buffer) {
			printWriter.println(s);
		}
		Buffer.clear();
		printWriter.flush();
	}
	
	void log(String s) {
		if(fileWriter == null) {
			Buffer.add(s);
		} else {
			printWriter.println(s);
			printWriter.flush();
		}
	}
	
	boolean isInitialized() {
		return fileWriter != null;
	}
}