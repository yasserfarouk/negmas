package genius.core.qualitymeasures;

/**
 * Simple object used to store the information of a scenario.
 * 
 * @author Mark Hendrikx
 */
public class ScenarioInfo {
	String domain;
	String prefProfA;
	String prefProfB;

	public ScenarioInfo(String domain) {
		this.domain = domain;
	}

	public String getDomain() {
		return domain;
	}

	public void setDomain(String domain) {
		this.domain = domain;
	}

	public String getPrefProfA() {
		return prefProfA;
	}

	public void setPrefProfA(String prefProfA) {
		this.prefProfA = prefProfA;
	}

	public String getPrefProfB() {
		return prefProfB;
	}

	public void setPrefProfB(String prefProfB) {
		this.prefProfB = prefProfB;
	}

}