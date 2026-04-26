package agents.anac.y2011.ValueModelAgent;

public class ValueDecrease{
	private double decrease;
	//well its not real std deviation, its
	//an abstraction, but is similar some what to weighted average
	private double avDev;
	private double reliabilty;
	private int lastSent;
	private double highestReliability;
	
	
	ValueDecrease(double val, double rel,double dev){
		avDev = dev;
		reliabilty = rel;
		decrease = val;
		lastSent=-1;
		highestReliability=rel;
	}
	public double getDecrease(){
		return decrease;
	}
	public void forceChangeDecrease(double newDecrease){
		decrease = newDecrease;
	}
	public double getReliabilty(){
		return reliabilty;
	}
	public double getMaxReliabilty(){
		return reliabilty;
	}
	public double getDeviance(){
		return avDev;
	}
	public int lastSent(){
		return lastSent;
	}
	public void sent(int bidIndex){
		lastSent = bidIndex;
	}
	//if the value is 100% reliable and the change is also very reliable
	//what chunk  of the value should be given to the new value
	//the actual effect of the new value is dependent on the reliabilty
	//of both.
	static private final double tempralDifferenceGamma = 0.1;
	private double sq(double x){return x*x;}
	public void updateWithNewValue(double newVal,double newReliability){
		if(reliabilty!=0.02){
			double newChunk =  newReliability*(1-reliabilty);
			double temporalC = reliabilty*tempralDifferenceGamma*newReliability;
			newChunk+=temporalC;
			double oldChunk =  reliabilty-temporalC;
			double sumChunk = newChunk+oldChunk;
			//newChunk/=sumChunk;
			//oldChunk/=sumChunk;
			double newDecrease = (newChunk*newVal+oldChunk*decrease)/sumChunk;
			double change = Math.abs(newDecrease-decrease);
			//this is a simplification of a real deviance calculation
			double newDev = Math.sqrt((oldChunk/2*sq(avDev+change)
							+oldChunk/2*sq(avDev-change)
							+newChunk*sq(newVal-newDecrease))/sumChunk);
			double temp = 1-change/(2*newDev);
			reliabilty = oldChunk * (temp>0.2?temp:0.2);
			temp = 1-(Math.abs(newDecrease-newVal)/(2*newDev));
			reliabilty +=newChunk * (temp>0?temp:0);
			decrease = newDecrease;
			avDev = newDev;
			if(highestReliability<reliabilty){
				highestReliability = reliabilty;
			}
		}
		//if this is the first time we really got this value than
		//the original value is meaningless
		else{
			decrease = newVal;
			reliabilty = newReliability;
			avDev = 0.03;
		}
	}
}