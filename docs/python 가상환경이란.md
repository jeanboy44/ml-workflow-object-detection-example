# 가상 환경(Virtual Environments) 파헤치기

**저자:** Lukas Waymann  
**날짜:** 2018-02-13  
**원문:** [Virtual Environments Demystified](https://meribold.org/python/2018/02/13/virtual-environments-9487/)

---

다음은 어떤 방식으로든 가상 환경을 생성하거나 관리하는 데 도움을 주기 위한 프로그램들의 (전부는 아니지만) 목록입니다:

> Hatch, VirtualEnvManager, autoenv, fades, inve, pew, pipenv, pyenv-virtualenv, pyenv-virtualenvwrapper, pyenv, pyvenv, rvirtualenv, tox, v, venv, vex, virtual-python, virtualenv-burrito, virtualenv-mv, virtualenv, virtualenvwrapper-win, virtualenvwrapper, workingenv

분명히 이 모든 것을 제대로 파악하는 건 정말 어려운 일임에 틀림없습니다. 저는 수천 줄의 파이썬 코드를 작성해봤음에도 불구하고 우리가 여기서 정확히 어떤 문제를 해결하려고 하는지조차 모르고 있었고, 미묘하게 다른 이름을 가진 관련 프로그램들이 너무 많아 그동안 이에 대해 읽어보는 것조차 꺼려왔으니, 제가 바보인 것이 분명합니다.

그렇다면 **가상 환경**이란 무엇일까요? 공식 문서의 튜토리얼에서는 다음과 같이 설명합니다:

> 특정 버전의 파이썬 설치와 여러 추가 패키지를 포함하는 독립적인(self-contained) 디렉터리 트리입니다.

파이썬 인터프리터가 있는 디렉터리라고요? 충분히 쉽네요.

```bash
$mkdir virtual_env$ cp /bin/python3 virtual_env/

```

한번 확인해 봅시다. 디렉터리인가요? 네. 파이썬 설치가 포함되어 있나요? 네. 여러 추가 패키지가 포함되어 있나요? 0개도 숫자니까요! (체크.) 특정 버전인가요? 음...

```bash
$cd virtual_env/$ ./python3 --version
Python 3.6.3

```

이 정도면 될 것 같습니다. 그런데 정말 **독립적(self-contained)** 일까요? 자기 자신을 포함하고 있지는 않은데...

농담은 제쳐두고, 우리의 디렉터리가 파이썬에 가상 환경을 위한 표준 메커니즘을 통합한 제안인 [PEP 405](https://www.python.org/dev/peps/pep-0405/)에서 명시한 가상 환경이 되기 위해 빠진 것은 딱 두 가지뿐입니다.

1. `home = /usr/bin`이라는 줄을 포함하는 `pyvenv.cfg`라는 이름의 파일
2. `lib/python3.6/site-packages` 하위 디렉터리

(두 경로 모두 OS에 따라 다르며, 두 번째 경로는 사용되는 파이썬 버전에 따라서도 달라집니다.)

```bash
$echo 'home = /usr/bin' > pyvenv.cfg$ mkdir -p lib/python3.6/site-packages

```

또한 파이썬 바이너리를 `bin` 하위 디렉터리로 옮기겠습니다.

```bash
$ mkdir bin && mv python3 bin/

```

좋습니다. 이제 형식적으로 가상 환경의 자격을 갖춘 디렉터리가 생겼습니다:

```bash
$ tree --noreport .
├── bin
│   └── python3
├── lib
│   └── python3.6
│       └── site-packages
└── pyvenv.cfg

```

이제 다음 질문으로 넘어갑니다.

## 요점이 무엇인가요?

우리가 복사한 파이썬 바이너리를 실행할 때, `pyvenv.cfg` 파일은 시작 과정에서 일어나는 일을 변경합니다. `home` 키가 존재하면 파이썬은 이 바이너리가 가상 환경에 속해 있다는 것을 알게 되고, 키의 값(`/usr/bin`)은 표준 라이브러리를 포함한 완전한 파이썬 설치가 어디에 있는지 알려줍니다.

핵심은 `./lib/python3.6/site-packages`가 모듈 검색 경로의 일부가 된다는 것입니다. 즉, 우리는 이제 이 위치에 패키지를 설치할 수 있으며, 특히 동일한 시스템에 있는 다른 파이썬 프로그램의 의존성과 충돌할 수 있는 특정 버전을 설치할 수 있다는 점이 중요합니다.

예를 들어, 프로젝트에 `left-pad` 버전 0.0.3이 정확히 필요하다면:

```bash
$ pip3 install -t lib/python3.6/site-packages/ left-pad==0.0.3

```

이제 다음은 작동할 것입니다:

```bash
$ ./bin/python3 -c 'import left_pad'

```

반면, 의도한 대로 다음은 `ModuleNotFoundError`를 발생시켜야 합니다:

```bash
$ python3 -c 'import left_pad'

```

동일한 시스템의 다른 프로젝트는 이 프로젝트와 간섭 없이 자체 가상 환경에 다른 버전의 `left-pad`를 가질 수 있습니다.

## 가상 환경 생성을 위한 표준 도구

실제로 사람들은 가상 환경을 수동으로 생성하지 않습니다. 다시 위의 겁나게 긴 도구 목록으로 돌아가게 되죠. 다행히도 그중 하나는 다른 것들과 다릅니다. 왜냐하면 표준 라이브러리의 일부로 파이썬과 함께 제공되기 때문입니다: 바로 `venv`입니다.

가장 간단한 형태로서, `venv`는 다음과 같이 가상 환경을 생성하는 데 사용됩니다:

```bash
$ python3 -m venv virtual_env

```

이 명령은 `virtual_env` 디렉터리를 생성하고 파이썬 인터프리터도 복사하거나 심볼릭 링크를 생성합니다:

```bash
$cd virtual_env$ find -name python3
./bin/python3

```

또한 다른 많은 것들도 복사합니다. 제 경우 89개 하위 디렉터리에 650개 파일, 총 약 10 MiB 정도가 생성되었습니다. 그 파일 중 하나는 `pip` 바이너리이며, 이를 사용하여 추가 명령줄 인수 없이 가상 환경에 패키지를 설치할 수 있습니다:

```bash
$ ./bin/pip install left-pad

```

`venv` 사용법과 "activate" 스크립트 같은 선택적인 마법에 대해서는 [파이썬 튜토리얼](https://docs.python.org/3/tutorial/venv.html)이나 [venv 문서](https://docs.python.org/3/library/venv.html)에서 더 읽어볼 수 있습니다. 이 글은 가상 환경이 실제로 무엇인지 요점만 정리하기 위한 것입니다.

## 요약

가상 환경은 파이썬 인터프리터, 인터프리터의 시작에 영향을 주는 특별한 `pyvenv.cfg` 파일, 그리고 일부 서드파티 파이썬 패키지를 포함하는 **디렉터리**입니다. 가상 환경에 설치된 파이썬 패키지들은 동일한 시스템의 다른 파이썬 애플리케이션과 간섭하지 않습니다. "가상 환경 생성을 위한 표준 도구"는 `venv`입니다.

## 부록: 타임라인

저는 Ian Bicking의 `non_root_python.py`가 가상 환경을 생성하는 최초의 도구로서 자격이 있다고 생각합니다. 이를 바탕으로 2005년 10월 버전 0.6a6에서 EasyInstall에 `virtual-python.py`가 추가되었습니다. 다음은 주요 사건을 요약한 타임라인입니다.

* **2005-10-17:** `virtual-python.py`가 EasyInstall에 추가됨.
* **2006-03-08:** Ian Bicking이 `virtual-python.py` 개선에 관한 "Working Environment Brainstorm"이라는 블로그 글 게시.
* **2006-03-15:** Ian Bicking이 `working-env.py` 발표.
* **2006-04-26:** Ian Bicking이 `working-env.py`의 개선된 버전인 `workingenv` 발표.
* **2007-09-14:** `virtualenv`의 첫 커밋.
* **2007-10-10:** Ian Bicking이 `virtualenv` 발표: "Workingenv는 죽었다, Virtualenv 만세!"
* **2009-10-24:** `virtual-python.py`가 EasyInstall에서 제거됨.
* **2011-06-13:** PEP 405 생성됨.
* **2012-05-25:** PEP 405가 Python 3.3 포함용으로 승인됨.
* **2012-09-29:** Python 3.3 릴리스, `venv`와 `pyvenv`가 표준 라이브러리의 일부가 됨.
* **2014-03-16:** Python 3.4 릴리스, `venv`가 이제 "생성된 모든 가상 환경에 pip를 설치하는 것을 기본값으로 함".
* **2015-09-13:** Python 3.5 릴리스. "이제 가상 환경 생성에 `venv` 사용이 권장됨."
* **2016-12-23:** Python 3.6 릴리스; "`pyvenv`는 Python 3.3 및 3.4에서 가상 환경 생성을 위한 권장 도구였으나, Python 3.6에서
deprecated(사용 중단 예정)됨."
