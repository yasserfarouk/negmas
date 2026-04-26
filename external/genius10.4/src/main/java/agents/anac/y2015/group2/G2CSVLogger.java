package agents.anac.y2015.group2;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

public class G2CSVLogger {
	
	FileWriter fileWriter = null;
	PrintWriter printWriter;
	String eol = System.getProperty("line.separator");
	
	ArrayList<ArrayList<Double>> Buffer = new ArrayList<ArrayList<Double>>();
	
	G2CSVLogger () {
		
		//Empty previous logs
		File dir = new File("./logs/model/");
		if(dir != null){
			for(File file: dir.listFiles()) file.delete();
		}
		dir = new File("./logs/real/");
		if(dir != null){
			for(File file: dir.listFiles()) file.delete();
		}
		
	}
	
	void init (int partyNumber, String name) {
		try {
			fileWriter = new FileWriter("./logs/"+name+"/party"+partyNumber+".csv");
			//printWriter = new PrintWriter(fileWriter);
		
			for(ArrayList<Double> list: Buffer) {
				for(double s:list){
					fileWriter.append(String.valueOf(s));
					fileWriter.append(',');	
				}
				fileWriter.append(eol);	
			}
			Buffer.clear();
			fileWriter.flush();
		} catch (IOException e) {
			System.out.println("Could not create log for party " + partyNumber);
		}
		
		
	}
	
	void log(ArrayList<Double> list) {
		if(fileWriter == null) {
			Buffer.add(list);
		} else {
			try {
				for(double s:list){
					fileWriter.append(String.valueOf(s));
					fileWriter.append(',');	
				}
				fileWriter.append(eol);
				fileWriter.flush();
			}catch (IOException e) {
				System.out.println("Could not add to log");
			}
		}
	}
	void log(Double d) {
		if(fileWriter == null) {
			ArrayList<Double> list = new ArrayList<Double>();
			list.add(d);
			Buffer.add(list);
		} else {
			try {
				
				fileWriter.append(String.valueOf(d));
				fileWriter.append(eol);
				fileWriter.flush();
			}catch (IOException e) {
				System.out.println("Could not add to log");
			}
		}
	}
	
	boolean isInitialized() {
		return fileWriter != null;
	}
	void close(){
		try {
			fileWriter.close();
		}catch (IOException e) {
			System.out.println("Could not close file");
		}
	}
}
