package negotiator.parties;

public class BoulwareNegotiationParty
		extends AbstractTimeDependentNegotiationParty {

	@Override
	public double getE() {
		return 0.2;
	}

	@Override
	public String getDescription() {
		return "Time-based agent";
	}
}
