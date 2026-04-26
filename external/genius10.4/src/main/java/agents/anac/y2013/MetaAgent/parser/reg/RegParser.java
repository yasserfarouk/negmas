package agents.anac.y2013.MetaAgent.parser.reg;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import agents.anac.y2013.MetaAgent.Parser.Type;

public class RegParser {

	private Type _type;
	private HashMap<String,Double> values;
	
	public RegParser(String fileName, Type type) {
		values=new HashMap<String,Double>();
		this._type=type;
		String [] lines=readFile(fileName);
		for (int i = 0; i < lines.length; i++) {
			parseLine(lines[i]);
		}
	}

	private void parseLine(String line) {
		String keys[]=line.split(",");
		keys[0]=keys[0].replace("\"","");
		for (int i = 0; i < keys.length; i++) {
			values.put(keys[0],Double.parseDouble(keys[1]));
		}
	}
	public double calcReg(HashMap<String,Double>map){
		double ans=0;
		for (String modifier : values.keySet()) {
			double value=values.get(modifier);
			if(modifier.compareTo("(Intercept)")==0){
				ans+=value;;
				continue;
			}
			double temp=value*map.get(modifier);
			ans+=temp;
		}
		switch(this._type){
		case LINREG:
			return ans;
			
		case LOGREG:
			return calcLog(ans);
		default:
			return -1;
		}
		
	}
	private double calcLog(double ans) {
		double res=-1*ans;
		res=Math.exp(res);
		res+=1;
		return 1/res;
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
		return text.split("\n");
	}
	
}
