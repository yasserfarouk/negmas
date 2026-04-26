package negotiator.boaframework.offeringstrategy.anac2011.valuemodelagent;

public class RealValuedecreaseProxy extends ValueDecrease {

	private double portion;
	private ValueDecrease worstScale;
	RealValuedecreaseProxy(ValueDecrease worstScale,double portion) {
		super(0, 0, 0);
		this.portion =portion;
		this.worstScale=worstScale;
	}
	@Override
	public double getDecrease(){
		return worstScale.getDecrease()*portion;
	}
	public double getReliabilty(){
		return worstScale.getReliabilty();
	}
	public double getDeviance(){
		return worstScale.getDeviance()*portion;
	}
	public int lastSent(){
		return worstScale.lastSent();
	}
	public void sent(int bidIndex){
		worstScale.sent(bidIndex);
	}
	public void updateWithNewValue(double newVal,double newReliability){
		//if portion is 0, than this is the best case scenario...
		//unless I add a way to turn the linear direction of the issue
		//there is nothing we can learn from newVal, and any other value
		//but 0 is considered a mistake
		if(portion>0){
			worstScale.updateWithNewValue(newVal/portion, newReliability);
		}
	}

}
