package agents.anac.y2013.MetaAgent.parser.nn;

public class Arrow {
	int sourceId;
	int destId;
	double value;
	public Arrow(int sourceId, int destId, double value) {
		this.sourceId = sourceId;
		this.destId = destId;
		this.value = value;
	}
	@Override
	public String toString() {
		return "Arrow [sourceId=" + sourceId + ", destId=" + destId
				+ ", value=" + value + "]";
	}
	
}
