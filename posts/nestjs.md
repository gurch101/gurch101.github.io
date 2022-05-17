# Nestjs

### Controllers

Create a controller by calling `nest g controller cats`

```js
import { Controller, Get } from '@nestjs/common';

@Controller('cats')
export class CatsController {
  @Get()
  findAll(@Query(): ListAllEntitiesDto): string {
    // when a request handler returns a JS object or array, it will automatically be serialized to json.
    // response status code is always 200 by default except for POST which uses 201.
    // if you need the underlying req/res object use @Req() req: Request/@Res() res: Response
    // redirect using @Redirect(url, 301)
    return 'hello';
  }

  @Get(':id')
  findOne(@Param() params): string {
    console.log(params.id);
    // alt could use @Param(':id') id: string to access directly
  }

  @Post
  async create(@Body() createCatDto: CreateCatDto) {
    /*
    dto's should be classes
    export class CreateCatDto {
      name: string;
      age: number;
      breed: string;
    }
    */
  }
}

// make nest aware of the controller in module class
@Module({
  controllers: [CatsController]
})
export class AppModule {}
```

### Providers

A provider is a class that can be injected as a dependency.

Create a provider by calling `nest g service cats`

```js
// application-scoped by default - instantiated and resolved at bootup and destroyed at shut down
@Injectible()
export class CatsService {
  private readonly cats: Cat[] = [];

  create(cat: Cat) {
    this.cats.push(cat);
  }

  findAll(): Cat[] {
    return this.cats;
  }
}

export interface Cat {
  name: string;
  age: number;
  breed: string;
}

// register the provider in the module
@Module({
  controllers: [CatsController],
  providers: [CatsService],
})
export class AppModule {}
```

Inject into the controller via the constructor

```js
constructor(private readonly catsService: CatsService) {}
```

### Modules

provide metadata that Nest uses to build and resolve the application graph. Create a module with `nest g module cats`.

```js
// if this module should be globally accessible, use @Global()
@Module({
  controllers: [CatsController],
  providers: [CatsService],
  exports: [CatsService], // share the instance of CatsService between several other modules. Other modules can now imports: [CatsService] to use
})
export class CatsModule {}

@Module({
  imports: [CatsModule],
})
export class AppModule {}
```

```bash
src
  cats
    dto
      create-cat.dto.ts
    interfaces
      cat.interface.ts
    cats.controller.ts
    cats.module.ts
    cats.service.ts
  app.module.ts
  main.ts
```

### Unit Testing

```js
import { Test } from "@nestjs/testing";
import { CatsController } from "./cats.controller";
import { CatsService } from "./cats.service";

describe("CatsController", () => {
  let catsController: CatsController;
  let catsService: CatsService;

  beforeEach(async () => {
    const moduleRef = await Test.createTestingModule({
      controllers: [CatsController],
      providers: [CatsService],
    }).compile();

    catsService = moduleRef.get < CatsService > CatsService;
    catsController = moduleRef.get < CatsController > CatsController;
  });

  describe("findAll", () => {
    it("should return an array of cats", async () => {
      const result = ["test"];
      jest.spyOn(catsService, "findAll").mockImplementation(() => result);

      expect(await catsController.findAll()).toBe(result);
    });
  });
});
```

### Middleware

A function that receives request, response, and a next callback that is called before the route handler.

```js
import { Injectable, NestMiddleware } from "@nestjs/common";
import { Request, Response, NextFunction } from "express";

@Injectable()
export class LoggerMiddleware implements NestMiddleware {
  use(req: Request, res: Response, next: NextFunction) {
    console.log("Request...");
    next();
  }
}

// in the module
import { NestModule } from "@nestjs/common";
export class AppModule implements NestModule {
  configure(consumer: MiddlewareConsumer) {
    consumer
      // can take multiple middleware functions
      .apply(LoggerMiddleware)
      // can take a string, multiple strings, or a RouteInfo object
      .exclude()
      // can take a string, multiple strings, a RouteInfo object, a controller class, or multiple controller classes
      .forRoutes("cats");
  }
}

// for global middleware, use app.use(middleware) in main.ts
```

Use functional middleware if your middleware doesn't need dependencies

```js
import { Request, Response, NextFunction } from "express";

export function logger(req: Request, res: Response, next: NextFunction) {
  console.log(`Request...`);
  next();
}
```

### E2E Testing

```js
import * as request from "supertest";
import { Test } from "@nestjs/testing";
import { CatsModule } from "../../src/cats/cats.module";
import { CatsService } from "../../src/cats/cats.service";
import { INestApplication } from "@nestjs/common";

describe("Cats", () => {
  let app: INestApplication;
  let catsService = { findAll: () => ["test"] };

  beforeAll(async () => {
    const moduleRef = await Test.createTestingModule({
      imports: [CatsModule],
    })
      .overrideProvider(CatsService)
      .useValue(catsService)
      .compile();

    app = moduleRef.createNestApplication();
    await app.init();
  });

  it(`/GET cats`, () => {
    return request(app.getHttpServer()).get("/cats").expect(200).expect({
      data: catsService.findAll(),
    });
  });

  afterAll(async () => {
    await app.close();
  });
});
```

### Exception Filters
