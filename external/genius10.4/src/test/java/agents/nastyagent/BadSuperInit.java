package agents.nastyagent;


public class BadSuperInit extends NastyAgent {

	public BadSuperInit() {
		throw new NullPointerException("just throwing for fun");
	}

}
