package negotiator.repository.boa;

import static org.junit.Assert.assertEquals;

import java.io.File;

import javax.xml.bind.JAXBException;

import org.junit.Test;

import genius.core.repository.boa.BoaRepository;

public class TestBoaRepo {
	@Test
	public void smokeTest() throws JAXBException {

		BoaRepository.loadRepository(new File("src/test/resources/boarepository.xml"));
	}

	@Test
	public void testLoad() throws JAXBException {
		BoaRepository repo = BoaRepository.loadRepository(new File("src/test/resources/boarepository.xml"));
		System.out.println(repo);

		assertEquals(31, repo.getBiddingStrategies().size());
		assertEquals(38, repo.getAcceptanceConditions().size());
		assertEquals(22, repo.getOpponentModels().size());
		assertEquals(6, repo.getOpponentModelStrategies().size());
	}

}
