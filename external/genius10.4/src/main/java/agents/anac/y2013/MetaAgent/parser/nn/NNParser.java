package agents.anac.y2013.MetaAgent.parser.nn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Vector;

public class NNParser {
	String []sections;
	HashMap<Integer,String>vars;
	Vector<Arrow>arrows;
	int length;
	int lastNodeID;
	public NNParser(String fileName) {
		sections=readFile(fileName);
		for (int i = 0; i < sections.length; i++) {
			sections[i]=sections[i].trim();
		}
		parseVars(sections[1]);
		arrows=new Vector<Arrow>();
		parseArrows(sections[2].split("\n"),sections[3].split("\n"));
	}
	
	private void parseArrows(String[] weightsS, String[] sourcesS) {
		length++;
		HashMap<Integer,Double> weights=new HashMap<Integer,Double>();
		for (int i = 0; i < weightsS.length; i++) {
			String[] s2=weightsS[i].replace("\"", "").split(",");
			Integer id=Integer.parseInt(s2[0]);
			Double w=Double.parseDouble(s2[1]);
			weights.put(id,w);
		}
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < length; j++) {
				String[] s2=sourcesS[i*length+j].replace("\"", "").split(",");
				int from=Integer.parseInt(s2[1]);
				Integer wID=Integer.parseInt(s2[0]);
				double w=weights.get(wID);
				arrows.add(new Arrow(from, i+12, w));
			}
		}
		int i=length*4;
		
		for (int j = 0; j <5; j++) {
			String[] s2=sourcesS[i+j].replace("\"", "").split(",");
			int from=Integer.parseInt(s2[1]);
			Integer wID=Integer.parseInt(s2[0]);
			double w=weights.get(wID);
			arrows.add(new Arrow(from, lastNodeID, w));
		}
	}

	private void parseVars(String varsString) {
		vars=new HashMap<Integer,String>();
		String [] varsS=varsString.split("\n");
		length=varsS.length;
		for (int i = 0; i < varsS.length; i++) {
			parseVar(varsS[i]);
		}
		int length=varsS.length;
		for (int i = 1; i < 6; i++) {
			vars.put(i+length, "V"+(i+length));
		}
		lastNodeID=vars.size();
	}

	private void parseVar(String s) {
		s=s.replace("\"","");
		String []s2=s.split(",");
		Integer id=Integer.parseInt(s2[0]);
		String name=s2[1];
		vars.put(id,name);
	}

	private String[] readFile(String fileName) {
		String text="";
		try {
		    	BufferedReader br = new BufferedReader(new FileReader(fileName));
		        StringBuilder sb = new StringBuilder();
		        String line = br.readLine();
		        while (line != null) {
		            sb.append(line);
		            sb.append("\n");
		            line = br.readLine();
		        }
		         text= sb.toString();
		        br.close();
		    }catch(Exception e){
		    	e.printStackTrace();
		    }
		return text.trim().split("\"x\"");
	}
}
