package agents.nastyagent;

public class ThrowInConstructor extends NastyAgent {

	public ThrowInConstructor() {
		throw new RuntimeException("just throwing in constructor for fun");
	}

}
